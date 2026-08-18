# Runbook

## `model.max_model_len`

After the diagnosis GPU boots, pick `max_model_len` from GPU fit, not by dropping long prompts to make the mix easier.

1. Start `vllm serve` with a trial `max_model_len`. For Qwen2.5-7B on ~48 GB, try **8192** first.
2. Watch VRAM and the serve log `num_gpu_blocks`. OOM -> lower `max_model_len`. Headroom and long Azure prompts would be dropped -> raise `max_model_len`.
3. Write that number into `config/pin.yaml` `model.max_model_len`. Then `timed_trace` / `window_policy` drop rows with prompt+output > `max_model_len` so the jsonl matches the engine.

## Pins

- `model.revision`: after first Hugging Face download, pin the commit if you want reproducible weights.
- `image.digest`: after first pull, pin from `docker image inspect` `RepoDigest`. Do not use `:latest`.
