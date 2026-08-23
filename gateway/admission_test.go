package main

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testGateway(t *testing.T, policy string, maxOutstanding int, upstream http.Handler) (*gateway, *httptest.Server) {
	t.Helper()
	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)
	parsed, err := url.Parse(up.URL)
	if err != nil {
		t.Fatal(err)
	}
	gw := newGateway(config{
		Upstream:       parsed,
		Policy:         policy,
		MaxOutstanding: maxOutstanding,
		Events:         &bytes.Buffer{},
	}, http.DefaultTransport)
	return gw, up
}

// assertOverloaded checks status 503 and the exact overloaded JSON body.
func assertOverloaded(t *testing.T, resp *http.Response) {
	t.Helper()
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	want := []byte(`{"error":"overloaded"}`)
	if !bytes.Equal(body, want) {
		t.Fatalf("body=%q want %q", body, want)
	}
}

func waitOutstanding(t *testing.T, gw *gateway, want int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if gw.Outstanding() == want {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("outstanding=%d want %d", gw.Outstanding(), want)
}

func TestAcceptedIncrementsOnce(t *testing.T) {
	var inFlight int32
	started := make(chan struct{})
	release := make(chan struct{})
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&inFlight, 1)
		close(started)
		<-release
		_, _ = w.Write([]byte("ok"))
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)

	done := make(chan *http.Response, 1)
	go func() {
		resp, err := http.Get(srv.URL + "/v1/completions")
		if err != nil {
			t.Error(err)
			done <- nil
			return
		}
		done <- resp
	}()
	<-started
	if got := gw.Outstanding(); got != 1 {
		t.Fatalf("outstanding during request: %d", got)
	}
	_, accepted, _, _ := gw.snapshot()
	if accepted != 1 {
		t.Fatalf("accepted_total=%d", accepted)
	}
	close(release)
	resp := <-done
	if resp == nil {
		t.Fatal("no response")
	}
	resp.Body.Close()
	waitOutstanding(t, gw, 0)
	_, accepted, _, _ = gw.snapshot()
	if accepted != 1 {
		t.Fatalf("accepted_total after complete=%d", accepted)
	}
}

func TestNormalCompletionDecrementsOnce(t *testing.T) {
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("done"))
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	resp, err := http.Get(srv.URL + "/v1/completions")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != "done" {
		t.Fatalf("body=%q", body)
	}
	waitOutstanding(t, gw, 0)
}

func TestUpstreamErrorDecrementsOnce(t *testing.T) {
	gw := newGateway(config{
		Upstream: mustURL("http://127.0.0.1:1"),
		Policy:   policyUnlimited,
	}, http.DefaultTransport)
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	resp, err := http.Get(srv.URL + "/v1/completions")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	waitOutstanding(t, gw, 0)
	_, _, _, upstreamErrors := gw.snapshot()
	if upstreamErrors != 1 {
		t.Fatalf("upstream_errors=%d", upstreamErrors)
	}
}

func TestMidStreamUpstreamErrorDecrementsOnce(t *testing.T) {
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("no hijack")
		}
		conn, buf, err := hj.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\npartial")
		_ = buf.Flush()
		conn.Close()
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	resp, err := http.Get(srv.URL + "/v1/completions")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	waitOutstanding(t, gw, 0)
}

func TestClientDisconnectDoesNotLeak(t *testing.T) {
	unblocked := make(chan struct{})
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
		close(unblocked)
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/v1/completions", nil)
	if err != nil {
		t.Fatal(err)
	}
	errCh := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
		errCh <- err
	}()
	waitOutstanding(t, gw, 1)
	cancel()
	select {
	case <-unblocked:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream did not see cancel")
	}
	waitOutstanding(t, gw, 0)
	select {
	case <-errCh:
	case <-time.After(2 * time.Second):
		t.Fatal("client request did not return")
	}
}

func TestRejectionNeverIncrementsOutstanding(t *testing.T) {
	var hits int32
	gw, _ := testGateway(t, policyStatic, 1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		time.Sleep(80 * time.Millisecond)
		_, _ = w.Write([]byte("ok"))
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)

	started := make(chan struct{})
	go func() {
		close(started)
		resp, err := http.Get(srv.URL + "/v1/completions")
		if err == nil {
			resp.Body.Close()
		}
	}()
	<-started
	waitOutstanding(t, gw, 1)
	resp, err := http.Get(srv.URL + "/v1/completions")
	if err != nil {
		t.Fatal(err)
	}
	assertOverloaded(t, resp)
	if gw.Outstanding() != 1 {
		t.Fatalf("reject changed outstanding to %d", gw.Outstanding())
	}
	_, accepted, rejected, _ := gw.snapshot()
	if accepted != 1 || rejected != 1 {
		t.Fatalf("accepted=%d rejected=%d", accepted, rejected)
	}
	if atomic.LoadInt32(&hits) != 1 {
		t.Fatalf("upstream hits=%d", hits)
	}
	waitOutstanding(t, gw, 0)
}

// TestStaticCapHoldsUnderConcurrentAdmits piles many admits on a held upstream.
func TestStaticCapHoldsUnderConcurrentAdmits(t *testing.T) {
	// Peak outstanding never exceeds the static cap M.
	const maxOutstanding = 8
	const concurrent = 64
	hold := make(chan struct{})
	var inUpstream int32
	var rejected int32
	gw, _ := testGateway(t, policyStatic, maxOutstanding, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&inUpstream, 1)
		<-hold
		_, _ = w.Write([]byte("ok"))
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)

	var peak int64
	stopPeak := make(chan struct{})
	go func() {
		for {
			select {
			case <-stopPeak:
				return
			default:
				n := gw.Outstanding()
				for {
					old := atomic.LoadInt64(&peak)
					if n <= old || atomic.CompareAndSwapInt64(&peak, old, n) {
						break
					}
				}
			}
		}
	}()

	var wg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Get(srv.URL + "/v1/completions")
			if err != nil {
				t.Error(err)
				return
			}
			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				t.Error(err)
				return
			}
			if resp.StatusCode == http.StatusServiceUnavailable {
				atomic.AddInt32(&rejected, 1)
				if !bytes.Equal(body, []byte(`{"error":"overloaded"}`)) {
					t.Errorf("reject body=%q", body)
				}
			}
		}()
	}

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt32(&inUpstream)+atomic.LoadInt32(&rejected) == int32(concurrent) {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	gotIn := atomic.LoadInt32(&inUpstream)
	gotRej := atomic.LoadInt32(&rejected)
	n := gw.Outstanding()
	close(stopPeak)
	polled := atomic.LoadInt64(&peak)
	if gotIn+gotRej != int32(concurrent) {
		close(hold)
		wg.Wait()
		t.Fatalf("decided in_upstream=%d rejected=%d want %d", gotIn, gotRej, concurrent)
	}
	if n > int64(maxOutstanding) {
		close(hold)
		wg.Wait()
		t.Fatalf("outstanding %d exceeds max %d", n, maxOutstanding)
	}
	if polled > int64(maxOutstanding) {
		close(hold)
		wg.Wait()
		t.Fatalf("polled peak %d exceeds max %d", polled, maxOutstanding)
	}
	if n != int64(maxOutstanding) {
		close(hold)
		wg.Wait()
		t.Fatalf("outstanding=%d want %d at cap", n, maxOutstanding)
	}

	close(hold)
	wg.Wait()
	waitOutstanding(t, gw, 0)
}

func TestSlowDownstreamReleasesOnUpstreamTerminate(t *testing.T) {
	payload := bytes.Repeat([]byte("tok "), 4096)
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	firstWrite := make(chan struct{})
	blockFlush := make(chan struct{})
	bw := &blockingWriter{header: make(http.Header), firstWrite: firstWrite, block: blockFlush}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"stream":true}`))
	done := make(chan struct{})
	go func() {
		gw.ServeHTTP(bw, req)
		close(done)
	}()
	select {
	case <-firstWrite:
	case <-time.After(2 * time.Second):
		t.Fatal("no first downstream write")
	}
	waitOutstanding(t, gw, 0)
	close(blockFlush)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler stuck after unblocking downstream")
	}
	if !bytes.Equal(bw.buf.Bytes(), payload) {
		t.Fatalf("body len=%d want %d", bw.buf.Len(), len(payload))
	}
}

func TestStreamingResponseIntact(t *testing.T) {
	chunks := []string{"data: {\"id\":1}\n\n", "data: {\"id\":2}\n\n"}
	gw, _ := testGateway(t, policyUnlimited, 0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("no flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			_, _ = io.WriteString(w, chunk)
			flusher.Flush()
		}
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/v1/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Join(chunks, "")
	if string(body) != want {
		t.Fatalf("body=%q want %q", body, want)
	}
}

func TestOverloadIs503(t *testing.T) {
	gw, _ := testGateway(t, policyStatic, 1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte("ok"))
	}))
	rel, _ := gw.acquire()
	defer rel()
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	resp, err := http.Get(srv.URL + "/v1/completions")
	if err != nil {
		t.Fatal(err)
	}
	assertOverloaded(t, resp)
	_, _, rejected, _ := gw.snapshot()
	if rejected != 1 {
		t.Fatalf("rejected_total=%d", rejected)
	}
}

func TestNoInternalWaitingQueue(t *testing.T) {
	hold := make(chan struct{})
	var served int32
	gw, _ := testGateway(t, policyStatic, 1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&served, 1)
		<-hold
		_, _ = w.Write([]byte("ok"))
	}))
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)

	firstDone := make(chan struct{})
	go func() {
		resp, err := http.Get(srv.URL + "/v1/completions")
		if err == nil {
			resp.Body.Close()
		}
		close(firstDone)
	}()
	waitOutstanding(t, gw, 1)
	start := time.Now()
	resp, err := http.Get(srv.URL + "/v1/completions")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	assertOverloaded(t, resp)
	if elapsed > 100*time.Millisecond {
		t.Fatalf("reject waited %s; gateway queued", elapsed)
	}
	close(hold)
	<-firstDone
	if atomic.LoadInt32(&served) != 1 {
		t.Fatalf("queued extra upstream hit: %d", served)
	}
}

func TestUnlimitedSanityMatchesDirect(t *testing.T) {
	body := `{"id":"cmpl-1","choices":[{"text":"hi"}]}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("path=%s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(upstream.Close)
	parsed, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	gw := newGateway(config{Upstream: parsed, Policy: policyUnlimited}, http.DefaultTransport)
	proxy := httptest.NewServer(gw)
	t.Cleanup(proxy.Close)

	direct, err := http.Post(upstream.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"hi","max_tokens":1}`))
	if err != nil {
		t.Fatal(err)
	}
	directBody, _ := io.ReadAll(direct.Body)
	direct.Body.Close()
	via, err := http.Post(proxy.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"hi","max_tokens":1}`))
	if err != nil {
		t.Fatal(err)
	}
	viaBody, _ := io.ReadAll(via.Body)
	via.Body.Close()
	if direct.StatusCode != via.StatusCode {
		t.Fatalf("status direct=%d via=%d", direct.StatusCode, via.StatusCode)
	}
	if !bytes.Equal(directBody, viaBody) {
		t.Fatalf("body mismatch direct=%q via=%q", directBody, viaBody)
	}
}

func TestMetricsAndEvents(t *testing.T) {
	buf := &bytes.Buffer{}
	var mu sync.Mutex
	events := writerFunc(func(p []byte) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return buf.Write(p)
	})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	t.Cleanup(up.Close)
	gw := newGateway(config{
		Upstream:       mustURL(up.URL),
		Policy:         policyStatic,
		MaxOutstanding: 1,
		Events:         events,
	}, http.DefaultTransport)
	rel, _ := gw.acquire()
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/v1/completions", nil)
	req.Header.Set("X-Request-Id", "req-1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertOverloaded(t, resp)
	rel()
	metrics, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	text, _ := io.ReadAll(metrics.Body)
	metrics.Body.Close()
	got := string(text)
	for _, name := range []string{
		"gateway_outstanding",
		"gateway_accepted_total",
		"gateway_rejected_total",
		"gateway_upstream_errors_total",
	} {
		if !strings.Contains(got, name) {
			t.Fatalf("metrics missing %s:\n%s", name, got)
		}
	}
	mu.Lock()
	logged := buf.String()
	mu.Unlock()
	if !strings.Contains(logged, `"event":"reject"`) || !strings.Contains(logged, "req-1") {
		t.Fatalf("events=%s", logged)
	}
}

type blockingWriter struct {
	header     http.Header
	status     int
	buf        bytes.Buffer
	firstWrite chan struct{}
	block      chan struct{}
	once       sync.Once
}

func (w *blockingWriter) Header() http.Header { return w.header }

func (w *blockingWriter) WriteHeader(status int) { w.status = status }

func (w *blockingWriter) Write(p []byte) (int, error) {
	w.once.Do(func() {
		if w.firstWrite != nil {
			close(w.firstWrite)
		}
		if w.block != nil {
			<-w.block
		}
	})
	return w.buf.Write(p)
}

func (w *blockingWriter) Flush() {}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

func mustURL(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	if err != nil {
		panic(err)
	}
	return parsed
}

