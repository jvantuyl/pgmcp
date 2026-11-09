package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

// ----- unit tests -----

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

func TestCleanSQL(t *testing.T) {
	cases := map[string]string{
		"SELECT 1":                              "SELECT 1",
		"SELECT 1;":                             "SELECT 1",
		"SELECT 1;\n\n":                         "SELECT 1",
		"```sql\nSELECT 1;\n```":                "SELECT 1",
		"```\nSELECT COUNT(*) FROM users;\n```": "SELECT COUNT(*) FROM users",
		"  sql\nSELECT 2 ":                      "SELECT 2",
		"To do this we query pg_class.\n\n```sql\nSELECT relname\nFROM pg_class\nLIMIT 5;\n```\n\nThis uses estimates.": "SELECT relname\nFROM pg_class\nLIMIT 5",
		"```sql\nSELECT 1\n``` then also ```\nSELECT 2;\n```":                                                           "SELECT 2",
	}
	for in, want := range cases {
		if got := cleanSQL(in); got != want {
			t.Fatalf("cleanSQL(%q) = %q, want %q", in, got, want)
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

func TestClassifyModelReply(t *testing.T) {
	cases := []struct {
		in       string
		declined bool
		sql      string
		reason   string
	}{
		{"SELECT 1", false, "SELECT 1", ""},
		{"with q as (select 1) select * from q", false, "with q as (select 1) select * from q", ""},
		{"(SELECT 1) UNION (SELECT 2)", false, "(SELECT 1) UNION (SELECT 2)", ""},
		{"-- count users\nSELECT COUNT(*) FROM users", false, "-- count users\nSELECT COUNT(*) FROM users", ""},
		{"/* c */ SELECT 1", false, "/* c */ SELECT 1", ""},
		{"CANNOT_ANSWER: no invoices table", true, "", "no invoices table"},
		{"  CANNOT_ANSWER:\n  nothing here  ", true, "", "nothing here"},
		{"There is no invoices table in the schema.", true, "", "There is no invoices table in the schema."},
		{"Selecting is impossible here", true, "", "Selecting is impossible here"},
		{"SELEC COUNT(*) FROM users", false, "SELEC COUNT(*) FROM users", ""},
		{"selec id from users limit 5", false, "selec id from users limit 5", ""},
		{"", true, "", ""},
	}
	for _, c := range cases {
		sql, declined, reason := classifyModelReply(c.in)
		if declined != c.declined || sql != c.sql || reason != c.reason {
			t.Errorf("classifyModelReply(%q) = (%q,%v,%q); want (%q,%v,%q)", c.in, sql, declined, reason, c.sql, c.declined, c.reason)
		}
	}
}

func TestNormalizeValueUUID(t *testing.T) {
	u := [16]byte{0x0a, 0x1b, 0x2c, 0x3d, 0x00, 0x01, 0x40, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0xab, 0xcd}
	want := "0a1b2c3d-0001-4000-8000-00000000abcd"
	if got := normalizeValue(u); got != want {
		t.Fatalf("uuid: got %v want %s", got, want)
	}
	got, ok := normalizeValue([][16]byte{u, u}).([]string)
	if !ok || len(got) != 2 || got[1] != want {
		t.Fatalf("uuid array: got %v", got)
	}
	if got := normalizeValue(int64(7)); got != int64(7) {
		t.Fatalf("other values must pass through, got %v", got)
	}
}

func TestChat_OmitsSamplingParamsWhenUnset(t *testing.T) {
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
	if _, err := s.chat(context.Background(), "sys", "user", 10); err != nil {
		t.Fatalf("chat: %v", err)
	}
	for _, k := range []string{"temperature", "top_p", "chat_template_kwargs"} {
		if _, present := got[k]; present {
			t.Errorf("%s must not be sent when unset, body=%v", k, got)
		}
	}

	s.cfg.Temperature, s.cfg.TopP, s.cfg.DisableThinking = ptr(0.2), ptr(0.9), true
	if _, err := s.chat(context.Background(), "sys", "user", 10); err != nil {
		t.Fatalf("chat: %v", err)
	}
	if got["temperature"] != 0.2 || got["top_p"] != 0.9 {
		t.Errorf("expected temperature 0.2 / top_p 0.9, body=%v", got)
	}
}

// ----- integration tests (need DATABASE_URL) -----

// mockRepairingOpenAI answers the first generation with SQL referencing a column that does not
// exist, answers the repair round (recognised by the failure text in the prompt) with correct
// SQL, and answers an explanation round with prose. It counts calls so tests can assert rounds.
func mockRepairingOpenAI(t *testing.T, repairWorks bool, calls *int32) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.WriteHeader(404)
			return
		}
		atomic.AddInt32(calls, 1)
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
			ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`
			TopP               *float64       `json:"top_p"`
			Temperature        *float64       `json:"temperature"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		sys, user := "", ""
		for _, m := range req.Messages {
			switch m.Role {
			case "system":
				sys = m.Content
			case "user":
				user = m.Content
			}
		}
		if v, ok := req.ChatTemplateKwargs["enable_thinking"]; !ok || v != false {
			t.Errorf("expected chat_template_kwargs.enable_thinking=false, got %v", req.ChatTemplateKwargs)
		}
		if req.TopP == nil || *req.TopP != 0.8 || req.Temperature == nil || *req.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7 / top_p 0.8, got %v / %v", req.Temperature, req.TopP)
		}

		content := "SELECT id, nonexistent_column FROM public.users LIMIT 5;"
		switch {
		case strings.Contains(user, "Question: prose-decline"):
			// Model ignores the sentinel and explains in prose; must not be executed.
			content = "There is no `invoices` table in the provided schema. The closest tables are users and orders."
		case strings.Contains(user, "Question: uuid-row"):
			content = "SELECT '0a1b2c3d-0001-4000-8000-00000000abcd'::uuid AS id, ARRAY['0a1b2c3d-0001-4000-8000-00000000abcd'::uuid] AS ids LIMIT 1"
		case strings.Contains(user, "Question: decline-on-repair") && strings.Contains(user, "FAILED when executed"):
			content = "CANNOT_ANSWER: users has no nonexistent_column and nothing reachable via FK provides it."
		case strings.Contains(sys, "helping debug"):
			content = "The users table has no nonexistent_column; use id and email from public.users instead."
		case strings.Contains(user, "FAILED when executed"):
			if !strings.Contains(user, "nonexistent_column") || !strings.Contains(user, "SQLSTATE 42703") {
				t.Errorf("repair prompt must include the failed SQL and SQLSTATE, got:\n%s", user)
			}
			if repairWorks {
				content = "```sql\nSELECT id, email FROM public.users ORDER BY id LIMIT 5;\n```"
			}
		}
		resp := map[string]any{
			"id": "mock", "object": "chat.completion", "model": "mock",
			"choices": []map[string]any{{
				"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": content},
			}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

func ptr[T any](v T) *T { return &v }

func repairTestConfig(mockURL string) Config {
	return Config{
		DatabaseURL:     os.Getenv("DATABASE_URL"),
		OpenAIKey:       "test",
		OpenAIBase:      mockURL + "/v1",
		OpenAIModel:     "mock",
		SchemaTTL:       1 * time.Minute,
		QueryTO:         10 * time.Second,
		LLMTO:           10 * time.Second,
		DisableThinking: true,
		SchemaMaxChars:  -1,
		Temperature:     ptr(0.7),
		TopP:            ptr(0.8),
		MaxRows:         50,
	}
}

func TestAsk_RepairRoundRecovers(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, true, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}

	res, out, err := srv.handleAsk(ctx, nil, askInput{Query: "List some users with their email"})
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}
	if res != nil && res.IsError {
		t.Fatalf("expected success after repair, got error result: %+v", out)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected 2 model calls (generate + repair), got %d", calls)
	}
	if len(out.Rows) == 0 {
		t.Fatalf("expected rows from repaired query, sql=%q", out.SQL)
	}
	if !strings.Contains(out.SQL, "email") || strings.HasSuffix(out.SQL, ";") {
		t.Fatalf("expected cleaned repaired sql, got %q", out.SQL)
	}
	if len(out.Attempts) != 1 || out.Attempts[0].SQLState != "42703" {
		t.Fatalf("expected one recorded failed attempt with SQLSTATE 42703, got %+v", out.Attempts)
	}
	if !strings.Contains(out.Note, "repaired after 1 failed attempt") {
		t.Fatalf("expected note to mention repair, got %q", out.Note)
	}
	if out.Error != "" || out.Explanation != "" {
		t.Fatalf("success must not carry error/explanation: %+v", out)
	}
}

func TestAsk_ExplanationRoundAfterRepairFails(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, false, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}

	res, out, err := srv.handleAsk(ctx, nil, askInput{Query: "List some users with their email"})
	if err != nil {
		t.Fatalf("handler must not return a Go error for a failed query: %v", err)
	}
	if res == nil || !res.IsError {
		t.Fatalf("expected IsError result, got %+v / %+v", res, out)
	}
	if atomic.LoadInt32(&calls) != 3 {
		t.Fatalf("expected 3 model calls (generate + repair + explain), got %d", calls)
	}
	if out.SQLState != "42703" || !strings.Contains(out.Error, "nonexistent_column") {
		t.Fatalf("expected the real database error, got sqlstate=%q error=%q", out.SQLState, out.Error)
	}
	if len(out.Attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %+v", out.Attempts)
	}
	if !strings.Contains(out.Explanation, "nonexistent_column") {
		t.Fatalf("expected explanation from third round, got %q", out.Explanation)
	}
	if !strings.Contains(out.Note, "failed after 2 attempts") {
		t.Fatalf("unexpected note %q", out.Note)
	}
	if len(out.Rows) != 0 {
		t.Fatalf("error result must not carry rows")
	}
}

func TestStream_RepairRoundRecovers(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, true, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	res, out, err := srv.handleStream(ctx, nil, streamInput{Query: "List some users with their email"})
	if err != nil || (res != nil && res.IsError) {
		t.Fatalf("expected success after repair: err=%v out=%+v", err, out)
	}
	if out.TotalRows == 0 || len(out.Attempts) != 1 {
		t.Fatalf("expected rows and one recorded attempt, got %+v", out)
	}
}

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

func TestAsk_ProseReplyIsNotExecuted(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, true, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	res, out, err := srv.handleAsk(ctx, nil, askInput{Query: "prose-decline"})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if res == nil || !res.IsError {
		t.Fatalf("expected error result, got %+v", out)
	}
	if atomic.LoadInt32(&calls) != 1 {
		t.Fatalf("prose reply must stop after 1 model call (no repair), got %d", calls)
	}
	if len(out.Attempts) != 0 || out.SQL != "" || len(out.Rows) != 0 {
		t.Fatalf("prose must not be executed: %+v", out)
	}
	if !strings.Contains(out.Explanation, "no `invoices` table") {
		t.Fatalf("explanation should carry the model's text, got %q", out.Explanation)
	}
	if !strings.Contains(out.Error, "could not translate") || !strings.Contains(out.Note, "model declined") {
		t.Fatalf("unexpected error/note: %q / %q", out.Error, out.Note)
	}
}

func TestAsk_RepairRoundMayDecline(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, true, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	res, out, err := srv.handleAsk(ctx, nil, askInput{Query: "decline-on-repair"})
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	if res == nil || !res.IsError {
		t.Fatalf("expected error result, got %+v", out)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected generate + repair, no explanation round; got %d calls", calls)
	}
	if len(out.Attempts) != 1 || out.Attempts[0].SQLState != "42703" {
		t.Fatalf("expected the one real failed attempt recorded, got %+v", out.Attempts)
	}
	if !strings.Contains(out.Explanation, "nothing reachable via FK") || strings.Contains(out.Explanation, "CANNOT_ANSWER") {
		t.Fatalf("explanation should be the sentinel's reason without the prefix, got %q", out.Explanation)
	}
	if !strings.Contains(out.Note, "model declined after 1 failed attempt") {
		t.Fatalf("unexpected note %q", out.Note)
	}
}

func TestAsk_UUIDsRenderAsText(t *testing.T) {
	db := mustPool(t)
	defer db.Close()
	resetSchema(t, db)

	var calls int32
	mock := mockRepairingOpenAI(t, true, &calls)
	defer mock.Close()

	ctx := context.Background()
	srv, err := newServer(ctx, repairTestConfig(mock.URL))
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	res, out, err := srv.handleAsk(ctx, nil, askInput{Query: "uuid-row"})
	if err != nil || (res != nil && res.IsError) {
		t.Fatalf("ask failed: err=%v out=%+v", err, out)
	}
	if len(out.Rows) != 1 {
		t.Fatalf("expected one row, got %+v", out.Rows)
	}
	const want = "0a1b2c3d-0001-4000-8000-00000000abcd"
	if out.Rows[0]["id"] != want {
		t.Fatalf("uuid column rendered as %#v, want %q", out.Rows[0]["id"], want)
	}
	ids, ok := out.Rows[0]["ids"].([]any)
	if !ok || len(ids) != 1 || ids[0] != want {
		t.Fatalf("uuid[] column rendered as %#v, want [%q]", out.Rows[0]["ids"], want)
	}
}
