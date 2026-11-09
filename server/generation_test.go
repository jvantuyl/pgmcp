package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

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
