package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

func TestTruncateSchema(t *testing.T) {
	txt := "TABLE a(x int)\nTABLE b(y int)\nTABLE c(z int)\nFK a(x) -> b(y)\n"

	if got := truncateSchema(txt, 0); got != txt {
		t.Fatalf("cap 0 must not truncate")
	}
	if got := truncateSchema(txt, -1); got != txt {
		t.Fatalf("negative cap must not truncate")
	}
	if got := truncateSchema(txt, len(txt)); got != txt {
		t.Fatalf("cap equal to length must not truncate")
	}

	// Cut in the middle of the third line: result must end after the second complete line.
	got := truncateSchema(txt, len("TABLE a(x int)\nTABLE b(y int)\nTABLE c("))
	if !strings.HasPrefix(got, "TABLE a(x int)\nTABLE b(y int)\n-- ...schema truncated: 2 more lines omitted") {
		t.Fatalf("expected line-boundary truncation with 2 omitted lines, got:\n%s", got)
	}
	if strings.Contains(got, "TABLE c(") {
		t.Fatalf("partial line leaked through: %s", got)
	}
}

func TestParseSchemaMaxChars(t *testing.T) {
	cases := []struct {
		in      string
		want    int
		wantErr bool
	}{
		{"", 0, false},
		{"  ", 0, false},
		{"none", -1, false},
		{"NONE", -1, false},
		{"50000", 50000, false},
		{" 123 ", 123, false},
		{"0", 0, true},
		{"-5", 0, true},
		{"lots", 0, true},
	}
	for _, c := range cases {
		got, err := parseSchemaMaxChars(c.in)
		if (err != nil) != c.wantErr {
			t.Fatalf("%q: err=%v wantErr=%v", c.in, err, c.wantErr)
		}
		if err == nil && got != c.want {
			t.Fatalf("%q: got %d want %d", c.in, got, c.want)
		}
	}
}

func TestModelContextLength(t *testing.T) {
	// sglang/vLLM style: alias is not directly gettable, but the only listed model reports max_model_len.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/v1/models":
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"served-name","object":"model","created":1,"owned_by":"sglang","max_model_len":196608}]}`))
		default:
			w.WriteHeader(404)
			_, _ = w.Write([]byte(`{"error":{"message":"not found"}}`))
		}
	}))
	defer srv.Close()

	llm := openai.NewClient(option.WithAPIKey("x"), option.WithBaseURL(srv.URL+"/v1"), option.WithMaxRetries(0))
	n, err := modelContextLength(context.Background(), llm, "alias")
	if err != nil {
		t.Fatalf("modelContextLength: %v", err)
	}
	if n != 196608 {
		t.Fatalf("got %d, want 196608", n)
	}

	// Plain OpenAI style: no max_model_len anywhere -> error, caller falls back.
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"alias","object":"model","created":1,"owned_by":"openai"},{"id":"other","object":"model","created":1,"owned_by":"openai"}]}`))
	}))
	defer srv2.Close()
	llm2 := openai.NewClient(option.WithAPIKey("x"), option.WithBaseURL(srv2.URL+"/v1"), option.WithMaxRetries(0))
	if _, err := modelContextLength(context.Background(), llm2, "alias"); err == nil {
		t.Fatalf("expected error when max_model_len is not reported")
	}
}

func TestCleanSQL(t *testing.T) {
	cases := map[string]string{
		"SELECT 1":                              "SELECT 1",
		"SELECT 1;":                             "SELECT 1",
		"SELECT 1;\n\n":                         "SELECT 1",
		"```sql\nSELECT 1;\n```":                "SELECT 1",
		"```\nSELECT COUNT(*) FROM users;\n```": "SELECT COUNT(*) FROM users",
		"  sql\nSELECT 2 ":                      "SELECT 2",
	}
	for in, want := range cases {
		if got := cleanSQL(in); got != want {
			t.Fatalf("cleanSQL(%q) = %q, want %q", in, got, want)
		}
	}
}

func ptr[T any](v T) *T { return &v }

func TestGenerateSQL_OmitsSamplingParamsWhenUnset(t *testing.T) {
	var got map[string]any
	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","object":"chat.completion","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"SELECT 1"}}]}`))
	}))
	defer mock.Close()

	s := &Server{
		llm:   openai.NewClient(option.WithAPIKey("x"), option.WithBaseURL(mock.URL+"/v1"), option.WithMaxRetries(0)),
		model: "m",
		cfg:   Config{LLMTO: 5 * time.Second},
	}
	if _, _, err := s.generateSQL(context.Background(), "q", "TABLE t(a int)", 10); err != nil {
		t.Fatalf("generateSQL: %v", err)
	}
	for _, k := range []string{"temperature", "top_p", "chat_template_kwargs"} {
		if _, present := got[k]; present {
			t.Errorf("%s must not be sent when unset, body=%v", k, got)
		}
	}

	s.cfg.Temperature, s.cfg.TopP, s.cfg.DisableThinking = ptr(0.2), ptr(0.9), true
	if _, _, err := s.generateSQL(context.Background(), "q", "TABLE t(a int)", 10); err != nil {
		t.Fatalf("generateSQL: %v", err)
	}
	if got["temperature"] != 0.2 || got["top_p"] != 0.9 {
		t.Errorf("expected temperature 0.2 / top_p 0.9, body=%v", got)
	}
	kw, _ := got["chat_template_kwargs"].(map[string]any)
	if kw == nil || kw["enable_thinking"] != false {
		t.Errorf("expected chat_template_kwargs.enable_thinking=false, body=%v", got)
	}
}

// ----- integration tests (need DATABASE_URL) -----

func TestSchemaSummaryIncludesCommentsAndOrdersTablesFirst(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	ctx := context.Background()
	if _, err := db.Exec(ctx, `COMMENT ON TABLE public.users IS 'People who can log in'; COMMENT ON COLUMN public.users.email IS 'Login
email, unique'`); err != nil {
		t.Fatalf("comment: %v", err)
	}
	schema, err := loadSchema(ctx, db)
	if err != nil {
		t.Fatalf("loadSchema: %v", err)
	}
	if !strings.Contains(schema, "-- People who can log in") {
		t.Fatalf("table comment missing from schema:\n%s", schema)
	}
	if !strings.Contains(schema, "{Login email, unique}") {
		t.Fatalf("column comment (with whitespace folded) missing from schema:\n%s", schema)
	}
	firstFK := strings.Index(schema, "\nFK ")
	lastTable := strings.LastIndex(schema, "\nTABLE ")
	if firstFK < 0 || lastTable < 0 || lastTable > firstFK {
		t.Fatalf("expected all TABLE lines before FK lines (lastTable=%d firstFK=%d)", lastTable, firstFK)
	}
}
