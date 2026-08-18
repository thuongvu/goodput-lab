#!/usr/bin/env bash
# Start vLLM, scrape /metrics while the engine is up, run vllm bench serve on
# a --timed-trace jsonl, write runs/<utc>_<id>/.
#
# discover: first proof; no --repeat. loaded: same commands; --repeat allowed
# (default if --mode omitted). The stage is which jsonl, not the mode.
#
#   ./src/goodput_lab/run.sh --timed-trace src/goodput_lab/data/slice.jsonl
#   ./src/goodput_lab/run.sh --mode discover --timed-trace src/goodput_lab/data/slice.jsonl
#   ./src/goodput_lab/run.sh --mode loaded --timed-trace ... --repeat 3
#   ./src/goodput_lab/run.sh --fit-max-model-len
#   ./src/goodput_lab/run.sh --download-model
#
# Every invoke downloads MODEL (cached if HF_HOME already has it). Empty pin
# max_model_len auto-searches before serve. --fit-max-model-len searches even
# if the pin already has a length.
#
# Boot is not data. --num-warmups is always set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${ROOT}/config/pin.yaml"
RUNS="${ROOT}/runs"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
SCRAPE_INTERVAL_MS="${SCRAPE_INTERVAL_MS:-200}"
NUM_WARMUPS="${NUM_WARMUPS:-5}"
CHUNK_HASH_SIZE=16
CLIENT_TASKSET="${CLIENT_TASKSET:-}"
IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-120}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
FIT_MAX_MODEL_LENS=(4096 8192 16384 32768)

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing pin config: ${CONFIG}" >&2
  exit 2
fi
if ! python3 -c "import goodput_lab" 2>/dev/null; then
  echo "goodput_lab is not installed" >&2
  exit 2
fi
eval "$(python3 -m goodput_lab.config "${CONFIG}")"
if [[ -z "${MODEL:-}" || -z "${DTYPE:-}" ]]; then
  echo "pin config must set model.name and model.dtype" >&2
  exit 2
fi
REVISION="${REVISION:-}"

MODE="loaded"
TIMED_TRACE=""
REPEAT=1
FIT_MAX_MODEL_LEN=0
DOWNLOAD_MODEL=0
DIFF_ARGS=()

# Print flags. --timed-trace is required for discover/loaded.
usage() {
  cat <<EOF
Usage: run.sh [--mode discover|loaded] --timed-trace JSONL [options]
       run.sh --fit-max-model-len
       run.sh --download-model

  --mode MODE              discover|loaded (default: loaded)
  --timed-trace JSONL      timed_trace jsonl (required for discover/loaded)
  --repeat N               N benches in one dir; wait idle between (loaded only)
  --fit-max-model-len      search max_model_len; write pin; no bench unless also --timed-trace
  --download-model         huggingface download of MODEL; then exit
  --diff ARGS...           extra vllm serve flags (must be last)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --timed-trace) TIMED_TRACE="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --fit-max-model-len) FIT_MAX_MODEL_LEN=1; shift ;;
    --download-model) DOWNLOAD_MODEL=1; shift ;;
    --diff) shift; DIFF_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODE" in
  discover|loaded) ;;
  *) echo "unknown mode: $MODE" >&2; usage; exit 2 ;;
esac
if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  if [[ -n "$TIMED_TRACE" || "$FIT_MAX_MODEL_LEN" -eq 1 || "$REPEAT" -gt 1 ]]; then
    echo "--download-model does not combine with --timed-trace, --fit-max-model-len, or --repeat" >&2
    exit 2
  fi
fi
if [[ "$FIT_MAX_MODEL_LEN" -eq 1 && "$REPEAT" -gt 1 ]]; then
  echo "--repeat does not apply to --fit-max-model-len" >&2
  exit 2
fi

FIT_ONLY=0
if [[ "$DOWNLOAD_MODEL" -eq 0 && -z "$TIMED_TRACE" ]]; then
  if [[ "$FIT_MAX_MODEL_LEN" -eq 1 || -z "${MAX_MODEL_LEN:-}" ]]; then
    FIT_ONLY=1
  else
    echo "--timed-trace JSONL required" >&2
    exit 2
  fi
fi
if [[ -n "$TIMED_TRACE" && ! -f "$TIMED_TRACE" ]]; then
  echo "--timed-trace JSONL required" >&2
  exit 2
fi
if [[ -n "$TIMED_TRACE" && "$MODE" == "discover" && "$REPEAT" -gt 1 ]]; then
  echo "--repeat does not apply to discover" >&2
  exit 2
fi

# huggingface-cli download of pin MODEL into HF_HOME (or the default cache).
download_model() {
  local dest="${HF_HOME:-default Hugging Face cache}"
  echo "downloading ${MODEL} to ${dest}"
  local -a cmd=()
  if command -v huggingface-cli >/dev/null 2>&1; then
    cmd=(huggingface-cli download "$MODEL")
  elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    cmd=(python3 -m huggingface_hub.commands.huggingface_cli download "$MODEL")
  else
    echo "missing huggingface-cli (huggingface_hub)" >&2
    exit 2
  fi
  if [[ -n "${REVISION}" ]]; then
    cmd+=(--revision "$REVISION")
  fi
  "${cmd[@]}"
}

# Fail if vllm or nvidia-smi is missing.
preflight() {
  if ! command -v vllm >/dev/null 2>&1; then
    echo "missing vllm on PATH" >&2
    exit 2
  fi
  if ! nvidia-smi >/dev/null; then
    echo "nvidia-smi failed" >&2
    exit 2
  fi
}

# Make runs/<utc>_<id>/ with metadata.json, pin.yaml copy, empty serve.log.
new_run_dir() {
  local id utc dir git_head
  id="$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
  utc="$(date -u +%Y%m%dT%H%M%SZ)"
  dir="${RUNS}/${utc}_${id}"
  mkdir -p "$dir"
  git_head="uncommitted"
  git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1 && git_head="$(git -C "$ROOT" rev-parse HEAD)"
  python3 -m goodput_lab.run_metadata "$dir" "$git_head" "$CHUNK_HASH_SIZE" "$MODE"
  cp "${CONFIG}" "$dir/"
  touch "${dir}/serve.log"
  echo "$dir"
}

# Make runs/fit_<utc>_<id>/ for max_model_len search logs.
new_fit_dir() {
  local id utc dir
  id="$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
  utc="$(date -u +%Y%m%dT%H%M%SZ)"
  dir="${RUNS}/fit_${utc}_${id}"
  mkdir -p "$dir"
  cp "${CONFIG}" "$dir/"
  touch "${dir}/serve.log"
  echo "$dir"
}

# Start vllm serve if the port is free; wait until /v1/models answers.
start_server() {
  if curl -fsS -m 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "${BASE_URL}/v1/models already answers; refuse to start (wrong occupant on the port)" >&2
    return 1
  fi
  nvidia-smi >"${RUN_DIR}/nvidia-smi.txt" 2>&1
  local -a argv=(
    serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --no-enable-prefix-caching
    --max-model-len "$MAX_MODEL_LEN"
  )
  if [[ -n "${REVISION}" ]]; then
    argv+=(--revision "$REVISION")
  fi
  echo "serve: vllm ${argv[*]} ${DIFF_ARGS[*]:-}" | tee "${RUN_DIR}/serve.cmd"
  nohup vllm "${argv[@]}" "${DIFF_ARGS[@]}" >"${RUN_DIR}/serve.log" 2>&1 &
  SERVE_PID=$!
  echo "$SERVE_PID" >"${RUN_DIR}/serve.pid"
  wait_ready
}

# Background scrape of /metrics into the run dir until cleanup.
start_scrape() {
  STOP_FILE="${RUN_DIR}/scrape.stop"
  rm -f "$STOP_FILE"
  python3 "${SCRIPT_DIR}/scrape_metrics.py" \
    --url "${BASE_URL}/metrics" \
    --interval-ms "$SCRAPE_INTERVAL_MS" \
    --out "${RUN_DIR}/metrics.jsonl" \
    --stop-file "$STOP_FILE" &
  SCRAPE_PID=$!
}

# Poll /v1/models until serve is up or READY_TIMEOUT_S.
wait_ready() {
  local t0
  t0="$(date +%s)"
  while true; do
    curl -fsS -m 2 "${BASE_URL}/v1/models" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "vllm serve died (pid ${SERVE_PID})" >&2
      return 1
    fi
    if (( "$(date +%s)" - t0 >= READY_TIMEOUT_S )); then
      echo "timeout waiting for ${BASE_URL}/v1/models" >&2
      return 1
    fi
    sleep 1
  done
}

# Poll until /v1/models no longer answers after a serve stop.
wait_port_free() {
  local t0
  t0="$(date +%s)"
  while curl -fsS -m 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; do
    if (( "$(date +%s)" - t0 >= 30 )); then
      echo "timeout waiting for ${BASE_URL} to free after serve stop" >&2
      return 1
    fi
    sleep 1
  done
}

# Poll /metrics until running and waiting are 0 or IDLE_TIMEOUT_S.
wait_idle() {
  local t0 running waiting
  t0="$(date +%s)"
  while true; do
    running="$(curl -fsS -m 2 "${BASE_URL}/metrics" 2>/dev/null \
      | awk '$1 ~ /^vllm:num_requests_running(\{|$)/ { print $NF; exit }' || echo "")"
    waiting="$(curl -fsS -m 2 "${BASE_URL}/metrics" 2>/dev/null \
      | awk '$1 ~ /^vllm:num_requests_waiting(\{|$)/ { print $NF; exit }' || echo "")"
    if [[ -n "$running" && -n "$waiting" ]]; then
      awk -v r="$running" -v w="$waiting" 'BEGIN { exit !((r+0)==0 && (w+0)==0) }' && return 0
    fi
    if (( "$(date +%s)" - t0 >= IDLE_TIMEOUT_S )); then
      echo "timeout waiting idle running=${running:-?} waiting=${waiting:-?}" >&2
      return 1
    fi
    sleep 1
  done
}

# One vllm bench serve; result JSON name is the argument.
bench_once() {
  local result_name="$1"
  local timed_trace_count
  timed_trace_count="$(python3 -m goodput_lab.n_match --count "$TIMED_TRACE")"
  local -a cmd=(vllm bench serve
    --model "$MODEL"
    --dataset-name timed_trace
    --dataset-path "$TIMED_TRACE"
    --num-prompts "$timed_trace_count"
    --no-oversample
    --self-timed
    --ignore-eos
    --save-result
    --save-detailed
    --num-warmups "$NUM_WARMUPS"
    --timed-trace-chunk-hash-size "$CHUNK_HASH_SIZE"
    --timed-trace-sec-multiplier 1  # jsonl timestamps already seconds
    --host "$HOST"
    --port "$PORT"
    --result-dir "$RUN_DIR"
    --result-filename "$result_name"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,99
  )
  if [[ -n "${CLIENT_TASKSET}" ]]; then
    cmd=(taskset -c "$CLIENT_TASKSET" "${cmd[@]}")
  fi
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

# Require bench JSON; compare prompt counts to the timed-trace via n_match.
check_bench_counts() {
  local bench_json="${RUN_DIR}/$1"
  if [[ ! -f "$bench_json" ]]; then
    echo "bench failed and no result JSON" >&2
    return 1
  fi
  python3 -m goodput_lab.n_match "$TIMED_TRACE" "$bench_json"
}

SERVE_PID=""
SCRAPE_PID=""
STOP_FILE=""

# Stop the current vllm serve pid and wait until it is gone.
kill_serve() {
  if [[ -z "${SERVE_PID}" ]]; then
    return 0
  fi
  if kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
  fi
  wait "$SERVE_PID" 2>/dev/null || true
  SERVE_PID=""
}

# Write last successful search length into config/pin.yaml.
write_fit_pin() {
  local value="$1"
  python3 -c 'from pathlib import Path; from goodput_lab.config import write_max_model_len; write_max_model_len(Path("'"${CONFIG}"'"), '"${value}"')'
  echo "wrote model.max_model_len: ${value} to ${CONFIG}"
}

# Try FIT_MAX_MODEL_LENS in order. Stop at first failure after a success.
search_max_model_len() {
  local last_ok="" len ready_ec
  RUN_DIR="$(new_fit_dir)"
  echo "fit dir: $RUN_DIR"
  for len in "${FIT_MAX_MODEL_LENS[@]}"; do
    echo "fit: trying max_model_len=${len}"
    if curl -fsS -m 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
      echo "${BASE_URL}/v1/models already answers; refuse to start (wrong occupant on the port)" >&2
      if [[ -n "$last_ok" ]]; then
        echo "fit: port still occupied; keeping ${last_ok}"
        break
      fi
      return 1
    fi
    MAX_MODEL_LEN="$len"
    set +e
    start_server
    ready_ec=$?
    set -e
    if [[ "$ready_ec" -eq 0 ]]; then
      echo "fit: ${len} ready"
      last_ok="$len"
      kill_serve
      if ! wait_port_free; then
        echo "fit: port did not free; keeping ${last_ok}"
        break
      fi
      continue
    fi
    kill_serve
    wait_port_free || true
    if [[ -z "$last_ok" ]]; then
      echo "fit: ${len} did not come up; not writing model.max_model_len" >&2
      MAX_MODEL_LEN=""
      return 1
    fi
    echo "fit: ${len} did not come up; keeping ${last_ok}"
    break
  done
  MAX_MODEL_LEN="$last_ok"
  write_fit_pin "$last_ok"
}

# Stop scrape then serve on exit.
cleanup() {
  local ec=$?
  if [[ -n "${SCRAPE_PID}" ]] && kill -0 "$SCRAPE_PID" 2>/dev/null; then
    [[ -n "$STOP_FILE" ]] && touch "$STOP_FILE"
    wait "$SCRAPE_PID" 2>/dev/null || true
  fi
  kill_serve
  exit "$ec"
}

download_model
if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  exit 0
fi

trap cleanup EXIT
preflight

if [[ "$FIT_MAX_MODEL_LEN" -eq 1 || -z "${MAX_MODEL_LEN:-}" ]]; then
  search_max_model_len
  if [[ "$FIT_ONLY" -eq 1 ]]; then
    echo "done"
    exit 0
  fi
fi

if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
  echo "MAX_MODEL_LEN unset. Choose L from GPU fit, write it in config/pin.yaml, then filter." >&2
  exit 2
fi

RUN_DIR="$(new_run_dir)"
echo "run dir: $RUN_DIR"

start_server
start_scrape
if ! kill -0 "$SCRAPE_PID" 2>/dev/null; then
  echo "scrape died" >&2
  exit 1
fi

i=1
while [[ "$i" -le "$REPEAT" ]]; do
  if [[ "$REPEAT" -gt 1 ]]; then
    echo "=== repeat $i / $REPEAT ==="
    if [[ "$i" -gt 1 ]]; then
      wait_idle
    fi
    RESULT_NAME="rep_${i}.json"
  else
    RESULT_NAME="bench.json"
  fi
  set +e
  bench_once "$RESULT_NAME"
  bench_ec=$?
  set -e
  if [[ "$bench_ec" -ne 0 ]]; then
    exit "$bench_ec"
  fi
  check_bench_counts "$RESULT_NAME"
  i=$((i + 1))
done
echo "done -> $RUN_DIR"
