// Tiny admission proxy in front of stock vLLM.
//
// outstanding increments once when a request is forwarded, and decrements
// exactly once when the upstream request or stream terminates. A slow
// downstream client does not hold the slot. There is no internal queue:
// reject immediately with 503, or forward.
//
// timed_trace / vllm bench serve talks to the OpenAI-compatible HTTP API
// (typically POST /v1/completions, often streamed). This proxy forwards
// that shape unchanged: method, path, query, headers, and body.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	policyUnlimited = "unlimited"
	policyStatic    = "static"
)

type config struct {
	Upstream       *url.URL
	Policy         string
	MaxOutstanding int
	Events         io.Writer
}

type gateway struct {
	cfg       config
	transport http.RoundTripper

	mu              sync.Mutex
	outstanding     int64
	acceptedTotal   int64
	rejectedTotal   int64
	upstreamErrors  int64
	eventMu         sync.Mutex
}

func main() {
	listen := flag.String("listen", ":8080", "gateway listen address")
	upstream := flag.String("upstream", "http://127.0.0.1:8000", "stock vLLM base URL")
	policy := flag.String("policy", "unlimited", "unlimited or static")
	maxOutstanding := flag.Int("max-outstanding", 0, "static admission cap; ignored when unlimited")
	eventsPath := flag.String("events", "", "optional JSONL admission event log")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Admission proxy for stock vLLM.\n\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	parsed, err := url.Parse(*upstream)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		log.Fatalf("invalid -upstream %q", *upstream)
	}
	if *policy != policyUnlimited && *policy != policyStatic {
		log.Fatalf("invalid -policy %q (unlimited or static)", *policy)
	}
	if *policy == policyStatic && *maxOutstanding < 1 {
		log.Fatal("static policy requires -max-outstanding >= 1")
	}

	var events io.Writer
	if *eventsPath != "" {
		f, err := os.OpenFile(*eventsPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			log.Fatalf("events file: %v", err)
		}
		defer f.Close()
		events = f
	}

	roundTripper := http.DefaultTransport.(*http.Transport).Clone()
	roundTripper.MaxIdleConnsPerHost = 64
	gw := newGateway(config{
		Upstream:       parsed,
		Policy:         *policy,
		MaxOutstanding: *maxOutstanding,
		Events:         events,
	}, roundTripper)

	log.Printf("listen %s policy=%s max_outstanding=%d upstream=%s", *listen, *policy, *maxOutstanding, parsed)
	if err := http.ListenAndServe(*listen, gw); err != nil {
		log.Fatal(err)
	}
}

func newGateway(cfg config, transport http.RoundTripper) *gateway {
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &gateway{cfg: cfg, transport: transport}
}

func (g *gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/metrics" && r.Method == http.MethodGet {
		g.writeMetrics(w)
		return
	}
	release, before := g.acquire()
	if release == nil {
		g.reject(w, r, before)
		return
	}
	g.forward(w, r, release, before)
}

func (g *gateway) Outstanding() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.outstanding
}

func (g *gateway) snapshot() (outstanding, accepted, rejected, upstreamErrors int64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.outstanding, g.acceptedTotal, g.rejectedTotal, g.upstreamErrors
}

// acquire takes one slot under g.mu. Returns nil when static and at cap.
func (g *gateway) acquire() (func(), int64) {
	g.mu.Lock()
	before := g.outstanding
	if g.cfg.Policy == policyStatic && g.outstanding >= int64(g.cfg.MaxOutstanding) {
		g.mu.Unlock()
		return nil, before
	}
	g.outstanding++
	g.acceptedTotal++
	g.mu.Unlock()
	return func() {
		g.mu.Lock()
		g.outstanding--
		g.mu.Unlock()
	}, before
}

func (g *gateway) reject(w http.ResponseWriter, r *http.Request, before int64) {
	g.mu.Lock()
	g.rejectedTotal++
	g.mu.Unlock()
	g.logEvent("reject", before, requestID(r))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = w.Write([]byte(`{"error":"overloaded"}`))
}

func (g *gateway) forward(w http.ResponseWriter, r *http.Request, release func(), before int64) {
	g.logEvent("admit", before, requestID(r))

	upReq := r.Clone(r.Context())
	upReq.RequestURI = ""
	upReq.URL.Scheme = g.cfg.Upstream.Scheme
	upReq.URL.Host = g.cfg.Upstream.Host
	upReq.Host = g.cfg.Upstream.Host

	resp, err := g.transport.RoundTrip(upReq)
	if err != nil {
		g.noteUpstreamError()
		release()
		http.Error(w, "upstream error", http.StatusBadGateway)
		return
	}

	stream := newUpstreamStream(resp.Body, release)
	defer stream.Close()

	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	for {
		chunk, err := stream.Next()
		if len(chunk) > 0 {
			if _, writeErr := w.Write(chunk); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

func (g *gateway) noteUpstreamError() {
	g.mu.Lock()
	g.upstreamErrors++
	g.mu.Unlock()
}

func (g *gateway) writeMetrics(w http.ResponseWriter) {
	outstanding, accepted, rejected, upstreamErrors := g.snapshot()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE gateway_outstanding gauge\n")
	fmt.Fprintf(w, "gateway_outstanding %d\n", outstanding)
	fmt.Fprintf(w, "# TYPE gateway_accepted_total counter\n")
	fmt.Fprintf(w, "gateway_accepted_total %d\n", accepted)
	fmt.Fprintf(w, "# TYPE gateway_rejected_total counter\n")
	fmt.Fprintf(w, "gateway_rejected_total %d\n", rejected)
	fmt.Fprintf(w, "# TYPE gateway_upstream_errors_total counter\n")
	fmt.Fprintf(w, "gateway_upstream_errors_total %d\n", upstreamErrors)
}

func (g *gateway) logEvent(event string, outstandingBefore int64, id string) {
	if g.cfg.Events == nil {
		return
	}
	row := map[string]any{
		"ts_unix_ms":         time.Now().UnixMilli(),
		"event":              event,
		"outstanding_before": outstandingBefore,
		"max_outstanding":    g.cfg.MaxOutstanding,
		"request_id":         id,
		"policy":             g.cfg.Policy,
	}
	line, err := json.Marshal(row)
	if err != nil {
		return
	}
	g.eventMu.Lock()
	defer g.eventMu.Unlock()
	_, _ = g.cfg.Events.Write(append(line, '\n'))
}

func requestID(r *http.Request) string {
	// Header.Get is case-insensitive; X-Request-Id matches X-Request-ID.
	return r.Header.Get("X-Request-Id")
}

func copyHeader(dst, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Connection") || strings.EqualFold(key, "Transfer-Encoding") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

// upstreamStream reads the upstream body as fast as it arrives. release runs
// on upstream EOF or error, even if the client is still flushing. Remaining
// chunks buffer in RAM per request if the downstream client stalls.
type upstreamStream struct {
	mu     sync.Mutex
	cond   *sync.Cond
	chunks [][]byte
	err    error
	closed bool
}

func newUpstreamStream(body io.ReadCloser, onTerminate func()) *upstreamStream {
	s := &upstreamStream{}
	s.cond = sync.NewCond(&s.mu)
	go func() {
		defer body.Close()
		buf := make([]byte, 32*1024)
		for {
			n, err := body.Read(buf)
			if n > 0 {
				chunk := append([]byte(nil), buf[:n]...)
				s.mu.Lock()
				if !s.closed {
					s.chunks = append(s.chunks, chunk)
					s.cond.Signal()
				}
				s.mu.Unlock()
			}
			if err != nil {
				onTerminate()
				s.mu.Lock()
				s.err = err
				s.cond.Broadcast()
				s.mu.Unlock()
				return
			}
		}
	}()
	return s
}

func (s *upstreamStream) Next() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.chunks) == 0 && s.err == nil {
		s.cond.Wait()
	}
	if len(s.chunks) > 0 {
		chunk := s.chunks[0]
		s.chunks = s.chunks[1:]
		return chunk, nil
	}
	return nil, s.err
}

func (s *upstreamStream) Close() {
	s.mu.Lock()
	s.closed = true
	s.cond.Broadcast()
	s.mu.Unlock()
}
