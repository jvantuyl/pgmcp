// server/main.go
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Build-time variables (set via ldflags)
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

const (
	defaultSchemaTTL    = 5 * time.Minute
	defaultQueryTimeout = 25 * time.Second
	defaultLLMTimeout   = 18 * time.Second
	defaultMaxRows      = 200
	maxModelTokens      = 2000
	maxExplainTokens    = 500
	// Schema cap fallback when the backend does not report the model's context length.
	defaultSchemaMaxChars = 18000
	// Share of the model context the schema summary may occupy; the rest is prompt, question and answer.
	schemaCapFraction   = 0.7
	approxCharsPerToken = 4
	maxRequestSize      = 1024 * 1024 // 1MB max request size
	maxQueryLength      = 10000       // Max query length in characters
	pageSize            = 50          // Default page size for pagination
	maxPagesAuto        = 10          // Max pages to auto-fetch
)

type Server struct {
	db     *pgxpool.Pool
	llm    openai.Client // value type
	model  string
	cache  *SchemaCache
	cfg    Config
	server *http.Server
}

type Config struct {
	DatabaseURL string
	OpenAIKey   string
	OpenAIModel string
	OpenAIBase  string
	SchemaTTL   time.Duration
	QueryTO     time.Duration
	// LLMTO bounds the SQL-generation call to the model; zero means defaultLLMTimeout.
	LLMTO time.Duration
	// DisableThinking asks reasoning models (Qwen3 and friends served via vLLM/sglang)
	// to skip their hidden chain-of-thought, which otherwise dominates latency.
	DisableThinking bool
	// SchemaMaxChars caps the schema summary shown to the model: >0 is a fixed cap, <0 means no cap,
	// 0 derives the cap from the model's context length at startup.
	SchemaMaxChars int
	// Temperature and TopP are sent only when set; nil leaves the backend's defaults in place.
	Temperature *float64
	TopP        *float64
	MaxRows     int
}

// Validate checks if the configuration is valid and returns detailed errors
func (c *Config) Validate() error {
	var errs []string

	if c.DatabaseURL == "" {
		errs = append(errs, "DATABASE_URL is required")
	}

	if c.MaxRows <= 0 {
		errs = append(errs, "MAX_ROWS must be greater than 0")
	} else if c.MaxRows > 10000 {
		errs = append(errs, "MAX_ROWS cannot exceed 10000 (too many rows could cause memory issues)")
	}

	if c.QueryTO < time.Second {
		errs = append(errs, "QUERY_TIMEOUT must be at least 1 second")
	} else if c.QueryTO > 5*time.Minute {
		errs = append(errs, "QUERY_TIMEOUT cannot exceed 5 minutes")
	}

	if c.SchemaTTL < 30*time.Second {
		errs = append(errs, "SCHEMA_TTL must be at least 30 seconds")
	} else if c.SchemaTTL > 24*time.Hour {
		errs = append(errs, "SCHEMA_TTL cannot exceed 24 hours")
	}

	if c.LLMTO != 0 {
		if c.LLMTO < time.Second {
			errs = append(errs, "LLM_TIMEOUT must be at least 1 second")
		} else if c.LLMTO > 5*time.Minute {
			errs = append(errs, "LLM_TIMEOUT cannot exceed 5 minutes")
		}
	}

	if c.Temperature != nil && (*c.Temperature < 0 || *c.Temperature > 2) {
		errs = append(errs, "LLM_TEMPERATURE must be between 0 and 2")
	}
	if c.TopP != nil && (*c.TopP <= 0 || *c.TopP > 1) {
		errs = append(errs, "LLM_TOP_P must be between 0 (exclusive) and 1")
	}

	if len(errs) > 0 {
		return fmt.Errorf("configuration validation failed:\n  - %s", strings.Join(errs, "\n  - "))
	}

	return nil
}

type SchemaCache struct {
	mu        sync.RWMutex
	txt       string
	expiresAt time.Time
	ttl       time.Duration
	maxChars  int // <= 0 means no cap
}

func (c *SchemaCache) Get(ctx context.Context, db *pgxpool.Pool) (string, error) {
	c.mu.RLock()
	if time.Now().Before(c.expiresAt) && c.txt != "" {
		defer c.mu.RUnlock()
		return c.txt, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	if time.Now().Before(c.expiresAt) && c.txt != "" {
		return c.txt, nil
	}
	txt, err := loadSchema(ctx, db)
	if err != nil {
		return "", err
	}
	txt = truncateSchema(txt, c.maxChars)
	c.txt = txt
	c.expiresAt = time.Now().Add(c.ttl)
	return c.txt, nil
}

// truncateSchema cuts on a line boundary so the model never sees a half-written table.
func truncateSchema(txt string, maxChars int) string {
	if maxChars <= 0 || len(txt) <= maxChars {
		return txt
	}
	cut := strings.LastIndex(txt[:maxChars], "\n")
	if cut <= 0 {
		cut = maxChars
	}
	omitted := strings.Count(strings.TrimSpace(txt[cut:]), "\n") + 1
	return txt[:cut] + fmt.Sprintf("\n-- ...schema truncated: %d more lines omitted (raise SCHEMA_MAX_CHARS or set it to \"none\")...", omitted)
}

// parseSchemaMaxChars interprets SCHEMA_MAX_CHARS: a positive integer is a fixed cap,
// "none" disables the cap, and an empty value means derive it from the model.
func parseSchemaMaxChars(v string) (int, error) {
	v = strings.TrimSpace(v)
	switch {
	case v == "":
		return 0, nil
	case strings.EqualFold(v, "none"):
		return -1, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("must be a positive integer or \"none\"")
	}
	return n, nil
}

// modelContextLength asks the backend for the model's context window. The OpenAI API has no
// such field; vLLM and sglang report it as max_model_len on /v1/models. The configured model
// name may be an alias, so fall back to listing models and matching, or to the only model served.
func modelContextLength(ctx context.Context, llm openai.Client, model string) (int, error) {
	ctxTO, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	extract := func(m *openai.Model) (int, bool) {
		// Extra (non-schema) fields are stored with an "invalid" status by the SDK, so Valid()
		// is false even when the raw JSON is a perfectly good number; inspect Raw() directly.
		f, ok := m.JSON.ExtraFields["max_model_len"]
		if !ok || f.Raw() == "" || f.Raw() == "null" {
			return 0, false
		}
		var n int
		if err := json.Unmarshal([]byte(f.Raw()), &n); err != nil || n <= 0 {
			return 0, false
		}
		return n, true
	}

	if m, err := llm.Models.Get(ctxTO, model); err == nil {
		if n, ok := extract(m); ok {
			return n, nil
		}
	}
	page, err := llm.Models.List(ctxTO)
	if err != nil {
		return 0, err
	}
	for i := range page.Data {
		if page.Data[i].ID == model {
			if n, ok := extract(&page.Data[i]); ok {
				return n, nil
			}
		}
	}
	if len(page.Data) == 1 {
		if n, ok := extract(&page.Data[0]); ok {
			return n, nil
		}
	}
	return 0, errors.New("backend does not report max_model_len for this model")
}

func loadSchema(ctx context.Context, db *pgxpool.Pool) (string, error) {
	// COMMENT ON table/column text is folded onto one line and shown as `-- text` after a TABLE
	// line and `{text}` after a column, so descriptions live in the database rather than in a
	// hand-maintained hints file. TABLE lines sort before FK lines so that a truncated summary
	// loses relationships before it loses columns.
	q := `
WITH cols AS (
  SELECT n.nspname AS schema, c.relname AS table, a.attname AS column,
         pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
         (SELECT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = c.oid AND contype='p' AND a.attnum = ANY(conkey)
         )) AS is_pk,
         regexp_replace(col_description(c.oid, a.attnum), '\s+', ' ', 'g') AS col_comment,
         regexp_replace(obj_description(c.oid, 'pg_class'), '\s+', ' ', 'g') AS tbl_comment
  FROM pg_attribute a
  JOIN pg_class c ON a.attrelid = c.oid
  JOIN pg_namespace n ON c.relnamespace = n.oid
  WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relkind='r' AND n.nspname NOT IN ('pg_catalog','information_schema')
),
fks AS (
  SELECT
    n1.nspname AS src_schema, c1.relname AS src_table, a1.attname AS src_column,
    n2.nspname AS dst_schema, c2.relname AS dst_table, a2.attname AS dst_column
  FROM pg_constraint co
  JOIN pg_class c1 ON co.conrelid=c1.oid
  JOIN pg_namespace n1 ON c1.relnamespace=n1.oid
  JOIN pg_class c2 ON co.confrelid=c2.oid
  JOIN pg_namespace n2 ON c2.relnamespace=n2.oid
  JOIN unnest(co.conkey) WITH ORDINALITY AS ck(attnum, pos) ON TRUE
  JOIN unnest(co.confkey) WITH ORDINALITY AS fk(attnum, pos) ON ck.pos=fk.pos
  JOIN pg_attribute a1 ON a1.attrelid=c1.oid AND a1.attnum=ck.attnum
  JOIN pg_attribute a2 ON a2.attrelid=c2.oid AND a2.attnum=fk.attnum
  WHERE co.contype='f'
),
lines AS (
  SELECT
    'TABLE '||cols.schema||'.'||
    CASE
      WHEN cols.table ~ '^[a-z_][a-z0-9_]*$' THEN cols.table
      ELSE '"' || cols.table || '"'
    END ||'('||
      string_agg(
        CASE
          WHEN cols.column ~ '^[a-z_][a-z0-9_]*$' THEN cols.column
          ELSE '"' || cols.column || '"'
        END ||' '||cols.data_type||CASE WHEN cols.is_pk THEN ' PRIMARY KEY' ELSE '' END
          ||COALESCE(' {'||cols.col_comment||'}', ''),
        ', ' ORDER BY cols.column
      )||
    ')'||COALESCE(' -- '||cols.tbl_comment, '') AS line
  FROM cols
  GROUP BY cols.schema, cols.table, cols.tbl_comment
  UNION ALL
  SELECT 'FK '||src_schema||'.'||
    CASE
      WHEN src_table ~ '^[a-z_][a-z0-9_]*$' THEN src_table
      ELSE '"' || src_table || '"'
    END ||'('||
    CASE
      WHEN src_column ~ '^[a-z_][a-z0-9_]*$' THEN src_column
      ELSE '"' || src_column || '"'
    END ||') -> '||dst_schema||'.'||
    CASE
      WHEN dst_table ~ '^[a-z_][a-z0-9_]*$' THEN dst_table
      ELSE '"' || dst_table || '"'
    END ||'('||
    CASE
      WHEN dst_column ~ '^[a-z_][a-z0-9_]*$' THEN dst_column
      ELSE '"' || dst_column || '"'
    END ||')'
  FROM fks
)
SELECT line FROM lines ORDER BY line LIKE 'FK %', line;`

	ctxTO, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	rows, err := db.Query(ctxTO, q)
	if err != nil {
		return "", err
	}
	defer rows.Close()
	var b strings.Builder
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			return "", err
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String(), rows.Err()
}

func mustConfig() Config {
	var warnings []string

	ttl := defaultSchemaTTL
	if v := os.Getenv("SCHEMA_TTL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			ttl = d
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid SCHEMA_TTL '%s': %v, using default %v", v, err, defaultSchemaTTL))
		}
	}

	qto := defaultQueryTimeout
	if v := os.Getenv("QUERY_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			qto = d
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid QUERY_TIMEOUT '%s': %v, using default %v", v, err, defaultQueryTimeout))
		}
	}

	lto := defaultLLMTimeout
	if v := os.Getenv("LLM_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			lto = d
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid LLM_TIMEOUT '%s': %v, using default %v", v, err, defaultLLMTimeout))
		}
	}

	disableThinking := false
	if v := os.Getenv("LLM_DISABLE_THINKING"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			disableThinking = b
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid LLM_DISABLE_THINKING '%s': must be true/false, using default false", v))
		}
	}

	schemaCap, err := parseSchemaMaxChars(os.Getenv("SCHEMA_MAX_CHARS"))
	if err != nil {
		warnings = append(warnings, fmt.Sprintf("invalid SCHEMA_MAX_CHARS '%s': %v, deriving cap from the model's context length", os.Getenv("SCHEMA_MAX_CHARS"), err))
		schemaCap = 0
	}

	var temperature, topP *float64
	if v := os.Getenv("LLM_TEMPERATURE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			temperature = &f
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid LLM_TEMPERATURE '%s': %v, leaving the backend default", v, err))
		}
	}
	if v := os.Getenv("LLM_TOP_P"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			topP = &f
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid LLM_TOP_P '%s': %v, leaving the backend default", v, err))
		}
	}

	mr := defaultMaxRows
	if v := os.Getenv("MAX_ROWS"); v != "" {
		if n, err := fmt.Sscanf(v, "%d", &mr); n == 1 && err == nil && mr > 0 {
			// Successfully parsed
		} else {
			warnings = append(warnings, fmt.Sprintf("invalid MAX_ROWS '%s': must be a positive integer, using default %d", v, defaultMaxRows))
			mr = defaultMaxRows
		}
	}

	cfg := Config{
		DatabaseURL:     envOrDie("DATABASE_URL"),
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		OpenAIModel:     envDefault("OPENAI_MODEL", "gpt-4o-mini"),
		OpenAIBase:      os.Getenv("OPENAI_BASE_URL"),
		SchemaTTL:       ttl,
		QueryTO:         qto,
		LLMTO:           lto,
		DisableThinking: disableThinking,
		SchemaMaxChars:  schemaCap,
		Temperature:     temperature,
		TopP:            topP,
		MaxRows:         mr,
	}

	// Print warnings
	for _, warning := range warnings {
		log.Warn().Msg(warning)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		log.Fatal().Err(err).Msg("invalid configuration")
	}

	return cfg
}

func envOrDie(k string) string {
	v := os.Getenv(k)
	if v == "" {
		log.Fatal().Msgf("missing required env %s", k)
	}
	return v
}
func envDefault(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func newServer(ctx context.Context, cfg Config) (*Server, error) {
	conf, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		return nil, err
	}
	conf.MinConns = 2
	conf.MaxConns = 8
	conf.MaxConnLifetime = 30 * time.Minute
	conf.MaxConnIdleTime = 5 * time.Minute
	conf.HealthCheckPeriod = 30 * time.Second
	conf.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol

	db, err := pgxpool.NewWithConfig(ctx, conf)
	if err != nil {
		return nil, err
	}

	// Test the connection to ensure it's valid
	if err := db.Ping(ctx); err != nil {
		db.Close()
		return nil, err
	}

	var opts []option.RequestOption
	if cfg.OpenAIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.OpenAIKey))
	}
	if cfg.OpenAIBase != "" {
		opts = append(opts, option.WithBaseURL(cfg.OpenAIBase))
	}
	llm := openai.NewClient(opts...) // value

	schemaCap := cfg.SchemaMaxChars
	switch {
	case schemaCap < 0:
		log.Info().Msg("schema summary cap disabled (SCHEMA_MAX_CHARS=none)")
	case schemaCap > 0:
		log.Info().Int("schema_max_chars", schemaCap).Msg("schema summary cap set explicitly")
	default:
		if n, err := modelContextLength(ctx, llm, cfg.OpenAIModel); err == nil {
			schemaCap = int(float64(n) * schemaCapFraction * approxCharsPerToken)
			log.Info().Int("max_model_len", n).Int("schema_max_chars", schemaCap).Msg("schema summary cap derived from model context length")
		} else {
			schemaCap = defaultSchemaMaxChars
			log.Warn().Err(err).Int("schema_max_chars", schemaCap).Msg("could not determine model context length; using fallback cap, set SCHEMA_MAX_CHARS to override")
		}
	}

	return &Server{
		db:    db,
		llm:   llm,
		model: cfg.OpenAIModel,
		cache: &SchemaCache{ttl: cfg.SchemaTTL, maxChars: schemaCap},
		cfg:   cfg,
	}, nil
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	log.Info().Msg("shutting down server gracefully")

	var errs []error

	// Shutdown HTTP server
	if s.server != nil {
		if err := s.server.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("HTTP server shutdown error: %w", err))
		}
	}

	// Close database connections
	if s.db != nil {
		s.db.Close()
		log.Info().Msg("database connections closed")
	}

	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}

	log.Info().Msg("server shutdown complete")
	return nil
}

// ---------- helpers ----------

func minNonZero(v, max int) int {
	if v <= 0 {
		return max
	}
	if v > max {
		return max
	}
	return v
}

// sanitizeInput sanitizes and validates user input
func sanitizeInput(input string) error {
	input = strings.TrimSpace(input)

	if len(input) == 0 {
		return errors.New("input cannot be empty")
	}

	if len(input) > maxQueryLength {
		return fmt.Errorf("input too long: %d characters (max %d)", len(input), maxQueryLength)
	}

	// Check for potentially malicious patterns
	suspicious := []string{
		"--", "/*", "*/", "xp_", "sp_", "exec", "execute",
		"union", "information_schema", "pg_catalog",
	}

	lowerInput := strings.ToLower(input)
	for _, pattern := range suspicious {
		if strings.Contains(lowerInput, pattern) {
			log.Warn().Str("pattern", pattern).Str("input", input).Msg("suspicious pattern detected in input")
		}
	}

	return nil
}

// auditLog logs security-relevant events
func auditLog(event, user, query, result string, success bool) {
	log.Info().
		Str("event", event).
		Str("user", user).
		Str("query", query).
		Str("result", result).
		Bool("success", success).
		Msg("audit_log")
}

// isExpensiveQuery detects potentially expensive query patterns
func isExpensiveQuery(sql string) bool {
	sqlLower := strings.ToLower(sql)

	// Detect expensive patterns (generic)
	expensivePatterns := []string{
		"cross join", // Cartesian products
		"left join",  // LEFT JOINs can be expensive
	}

	for _, pattern := range expensivePatterns {
		if strings.Contains(sqlLower, pattern) {
			return true
		}
	}

	// Count number of JOINs - more than 2 is expensive
	joinCount := strings.Count(sqlLower, " join ")
	return joinCount > 2
}

// simplifyExpensiveQuery rewrites expensive queries to be more performant
func simplifyExpensiveQuery(sql, originalQuery string) string {
	sqlLower := strings.ToLower(sql)

	// For queries with expensive JOINs, return a helpful error message instead
	if strings.Contains(sqlLower, "left join") || strings.Contains(sqlLower, "cross join") || strings.Count(sqlLower, " join ") > 2 {
		return "SELECT 'Query too complex - please try a simpler question or ask about individual tables' AS message LIMIT 1"
	}

	return sql // Return original if no simplification needed
}

func validateSQLBasic(sql string) error {
	sqlLower := strings.ToLower(sql)

	// Check for basic SQL structure
	if !strings.Contains(sqlLower, "select") {
		return fmt.Errorf("query must contain SELECT")
	}

	// Check for balanced parentheses
	openCount := strings.Count(sql, "(")
	closeCount := strings.Count(sql, ")")
	if openCount != closeCount {
		return fmt.Errorf("unbalanced parentheses: %d open, %d close", openCount, closeCount)
	}

	// Check for common syntax issues
	if strings.Contains(sqlLower, "select select") {
		return fmt.Errorf("duplicate SELECT keywords detected")
	}

	return nil
}

// requestSizeLimitMiddleware limits the size of incoming requests
func requestSizeLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestSize)
		next.ServeHTTP(w, r)
	})
}

// ---------- MCP tool handlers ----------

type askInput struct {
	Query     string `json:"query"`
	MaxRows   int    `json:"max_rows,omitempty"`
	DryRun    bool   `json:"dry_run,omitempty"`
	Page      int    `json:"page,omitempty"`       // Page number (0-based)
	PageSize  int    `json:"page_size,omitempty"`  // Results per page
	StreamAll bool   `json:"stream_all,omitempty"` // Auto-fetch all pages
}

type askOutput struct {
	SQL        string           `json:"sql"`
	Rows       []map[string]any `json:"rows,omitempty"`
	Note       string           `json:"note,omitempty"`
	Page       int              `json:"page,omitempty"`
	PageSize   int              `json:"page_size,omitempty"`
	TotalCount int              `json:"total_count,omitempty"`
	HasMore    bool             `json:"has_more"`
	NextPage   int              `json:"next_page,omitempty"`
	// Set when the query could not be answered: the last database error, its SQLSTATE, every
	// SQL attempt with its error, and the model's explanation of why it failed.
	Error       string       `json:"error,omitempty"`
	SQLState    string       `json:"sqlstate,omitempty"`
	Attempts    []sqlAttempt `json:"attempts,omitempty"`
	Explanation string       `json:"explanation,omitempty"`
}

// sqlAttempt records one generated query that failed to execute.
type sqlAttempt struct {
	SQL      string `json:"sql"`
	Error    string `json:"error"`
	SQLState string `json:"sqlstate,omitempty"`
}

type streamInput struct {
	Query    string `json:"query"`
	MaxPages int    `json:"max_pages,omitempty"` // Max pages to fetch (default 10)
	PageSize int    `json:"page_size,omitempty"` // Results per page (default 50)
}

type streamOutput struct {
	SQL         string             `json:"sql"`
	Pages       []streamPageOutput `json:"pages"`
	TotalRows   int                `json:"total_rows"`
	TotalPages  int                `json:"total_pages"`
	Note        string             `json:"note,omitempty"`
	Error       string             `json:"error,omitempty"`
	SQLState    string             `json:"sqlstate,omitempty"`
	Attempts    []sqlAttempt       `json:"attempts,omitempty"`
	Explanation string             `json:"explanation,omitempty"`
}

type streamPageOutput struct {
	Page int              `json:"page"`
	Rows []map[string]any `json:"rows"`
}

func (s *Server) handleAsk(ctx context.Context, req *mcp.CallToolRequest, in askInput) (*mcp.CallToolResult, askOutput, error) {
	start := time.Now()
	clientIP := "unknown" // MCP doesn't expose client IP directly

	log.Debug().Str("tool", "ask").Str("query", strings.TrimSpace(in.Query)).
		Int("max_rows", in.MaxRows).Bool("dry_run", in.DryRun).Int("page", in.Page).
		Int("page_size", in.PageSize).Bool("stream_all", in.StreamAll).Str("client_ip", clientIP).Msg("request")

	// Input sanitization and validation
	if err := sanitizeInput(in.Query); err != nil {
		auditLog("ask_input_validation_failed", clientIP, in.Query, err.Error(), false)
		log.Debug().Str("tool", "ask").Err(err).Msg("input validation failed")
		return nil, askOutput{}, err
	}

	schemaTxt, err := s.cache.Get(ctx, s.db)
	if err != nil {
		log.Debug().Str("tool", "ask").Err(err).Msg("schema load failed")
		return nil, askOutput{}, err
	}

	// Determine page size
	pageSize := minNonZero(in.PageSize, pageSize)
	if in.MaxRows > 0 {
		pageSize = minNonZero(in.MaxRows, pageSize)
	}

	if in.DryRun {
		sql, note, err := s.generateSQL(ctx, in.Query, schemaTxt, pageSize*10)
		if err != nil {
			auditLog("ask_sql_generation_failed", clientIP, in.Query, err.Error(), false)
			log.Debug().Str("tool", "ask").Err(err).Msg("sql generation failed")
			return nil, askOutput{}, err
		}
		if err := guardReadOnly(sql); err != nil {
			auditLog("ask_dry_run_guard_failed", clientIP, sql, err.Error(), false)
			log.Debug().Str("tool", "ask").Err(err).Dur("dur", time.Since(start)).Msg("dry-run guard failed")
			return nil, askOutput{SQL: sql, Note: note}, err
		}
		auditLog("ask_dry_run_success", clientIP, in.Query, sql, true)
		log.Debug().Str("tool", "ask").Dur("dur", time.Since(start)).Msg("dry-run ok")
		return nil, askOutput{SQL: sql, Note: note}, nil
	}

	// Automatically stream all results
	maxPages := 20 // Auto-stream up to 20 pages (1000 results with default page size)
	if in.MaxRows > 0 {
		maxPages = (in.MaxRows + pageSize - 1) / pageSize // Calculate pages needed
	}

	var (
		pages       []streamPageOutput
		totalRows   int
		executedSQL string
	)
	outcome := s.generateAndRun(ctx, in.Query, schemaTxt, pageSize*10, func(sql string) error {
		// Check for potentially expensive queries and simplify them
		if isExpensiveQuery(sql) {
			log.Warn().Str("sql", sql).Msg("potentially expensive query detected - simplifying")
			sql = simplifyExpensiveQuery(sql, in.Query)
			log.Info().Str("simplified_sql", sql).Msg("query simplified for performance")
		}
		// Validate SQL syntax before execution (basic check)
		if err := validateSQLBasic(sql); err != nil {
			log.Warn().Str("sql", sql).Err(err).Msg("generated SQL may have issues")
			// Continue anyway - let the database provide the real error
		}
		executedSQL = sql
		var err error
		pages, totalRows, err = s.runStreamingQuery(ctx, sql, maxPages, pageSize)
		return err
	})

	if outcome.Err != nil {
		msg, state := describeErr(outcome.Err)
		auditLog("ask_query_failed", clientIP, cmp.Or(outcome.SQL, in.Query), msg, false)
		log.Debug().Str("tool", "ask").Err(outcome.Err).Int("attempts", len(outcome.Attempts)).
			Dur("dur", time.Since(start)).Msg("query failed")
		return &mcp.CallToolResult{IsError: true}, askOutput{
			SQL:         outcome.SQL,
			Note:        outcome.failureNote(),
			Error:       msg,
			SQLState:    state,
			Attempts:    outcome.Attempts,
			Explanation: outcome.Explanation,
		}, nil
	}

	// Flatten all pages into single result
	var allRows []map[string]any
	for _, page := range pages {
		allRows = append(allRows, page.Rows...)
	}

	auditLog("ask_success", clientIP, in.Query, fmt.Sprintf("streamed %d rows across %d pages", totalRows, len(pages)), true)
	log.Debug().Str("tool", "ask").Int("total_rows", totalRows).Int("pages", len(pages)).
		Int("returned_rows", len(allRows)).Dur("dur", time.Since(start)).Msg("done")

	return nil, askOutput{
		SQL:      executedSQL,
		Rows:     allRows,
		Note:     fmt.Sprintf("%s (streamed %d pages)", outcome.Note, len(pages)),
		Attempts: outcome.Attempts,
	}, nil
}

type searchInput struct {
	Q     string `json:"q"`
	Limit int    `json:"limit,omitempty"`
}

type searchOutput struct {
	SQL  string           `json:"sql"`
	Rows []map[string]any `json:"rows"`
}

func (s *Server) handleSearch(ctx context.Context, req *mcp.CallToolRequest, in searchInput) (*mcp.CallToolResult, searchOutput, error) {
	start := time.Now()
	clientIP := "unknown" // MCP doesn't expose client IP directly

	log.Debug().Str("tool", "search").Str("q", strings.TrimSpace(in.Q)).Int("limit", in.Limit).Str("client_ip", clientIP).Msg("request")

	// Input sanitization and validation
	if err := sanitizeInput(in.Q); err != nil {
		auditLog("search_input_validation_failed", clientIP, in.Q, err.Error(), false)
		log.Debug().Str("tool", "search").Err(err).Msg("input validation failed")
		return nil, searchOutput{}, err
	}
	limit := minNonZero(in.Limit, 50)
	sql, err := s.buildSearchSQL(ctx, in.Q, limit)
	if err != nil {
		auditLog("search_sql_build_failed", clientIP, in.Q, err.Error(), false)
		log.Debug().Str("tool", "search").Err(err).Msg("build sql failed")
		return nil, searchOutput{}, err
	}
	log.Debug().Str("tool", "search").Str("sql", sql).Msg("generated sql")

	rows, err := s.runReadOnlyQuery(ctx, sql, limit)
	if err != nil {
		auditLog("search_query_failed", clientIP, sql, err.Error(), false)
		log.Debug().Str("tool", "search").Err(err).Dur("dur", time.Since(start)).Msg("query failed")
		return nil, searchOutput{SQL: sql}, err
	}
	auditLog("search_success", clientIP, in.Q, fmt.Sprintf("returned %d rows", len(rows)), true)
	log.Debug().Str("tool", "search").Int("row_count", len(rows)).Dur("dur", time.Since(start)).Msg("done")
	return nil, searchOutput{SQL: sql, Rows: rows}, nil
}

func (s *Server) handleStream(ctx context.Context, req *mcp.CallToolRequest, in streamInput) (*mcp.CallToolResult, streamOutput, error) {
	start := time.Now()
	clientIP := "unknown"

	log.Debug().Str("tool", "stream").Str("query", strings.TrimSpace(in.Query)).
		Int("max_pages", in.MaxPages).Int("page_size", in.PageSize).Str("client_ip", clientIP).Msg("request")

	// Input sanitization and validation
	if err := sanitizeInput(in.Query); err != nil {
		auditLog("stream_input_validation_failed", clientIP, in.Query, err.Error(), false)
		log.Debug().Str("tool", "stream").Err(err).Msg("input validation failed")
		return nil, streamOutput{}, err
	}

	schemaTxt, err := s.cache.Get(ctx, s.db)
	if err != nil {
		log.Debug().Str("tool", "stream").Err(err).Msg("schema load failed")
		return nil, streamOutput{}, err
	}

	// Parameters
	maxPages := minNonZero(in.MaxPages, maxPagesAuto)
	pageSize := minNonZero(in.PageSize, pageSize)

	var (
		pages     []streamPageOutput
		totalRows int
	)
	outcome := s.generateAndRun(ctx, in.Query, schemaTxt, pageSize*maxPages, func(sql string) error {
		var err error
		pages, totalRows, err = s.runStreamingQuery(ctx, sql, maxPages, pageSize)
		return err
	})

	if outcome.Err != nil {
		msg, state := describeErr(outcome.Err)
		auditLog("stream_query_failed", clientIP, cmp.Or(outcome.SQL, in.Query), msg, false)
		log.Debug().Str("tool", "stream").Err(outcome.Err).Int("attempts", len(outcome.Attempts)).
			Dur("dur", time.Since(start)).Msg("query failed")
		return &mcp.CallToolResult{IsError: true}, streamOutput{
			SQL:         outcome.SQL,
			Pages:       []streamPageOutput{},
			Note:        outcome.failureNote(),
			Error:       msg,
			SQLState:    state,
			Attempts:    outcome.Attempts,
			Explanation: outcome.Explanation,
		}, nil
	}

	totalPages := (totalRows + pageSize - 1) / pageSize // Ceiling division

	auditLog("stream_success", clientIP, in.Query, fmt.Sprintf("returned %d rows in %d pages", totalRows, len(pages)), true)
	log.Debug().Str("tool", "stream").Int("total_rows", totalRows).Int("pages", len(pages)).
		Dur("dur", time.Since(start)).Msg("done")

	return nil, streamOutput{
		SQL:        outcome.SQL,
		Pages:      pages,
		TotalRows:  totalRows,
		TotalPages: totalPages,
		Note:       outcome.Note,
		Attempts:   outcome.Attempts,
	}, nil
}

// ---------- SQL helpers ----------

var mutating = regexp.MustCompile(`(?is)\b(INSERT|UPDATE|DELETE|UPSERT|MERGE|ALTER|DROP|TRUNCATE|VACUUM|REINDEX|GRANT|REVOKE|CREATE|COPY|ROLLBACK|COMMIT|BEGIN|START|SAVEPOINT|RELEASE|SET)\b`)

func guardReadOnly(sql string) error {
	if mutating.MatchString(sql) {
		return fmt.Errorf("refusing to run non-read-only SQL")
	}
	// disallow multiple statements
	if strings.Count(sql, ";") > 0 && strings.TrimSpace(sql)[len(strings.TrimSpace(sql))-1] != ';' {
		return fmt.Errorf("multiple statements not allowed")
	}
	return nil
}

func (s *Server) runReadOnlyQuery(ctx context.Context, sql string, limit int) ([]map[string]any, error) {
	ctxTO, cancel := context.WithTimeout(ctx, s.cfg.QueryTO)
	defer cancel()
	conn, err := s.db.Acquire(ctxTO)
	if err != nil {
		return nil, err
	}
	defer conn.Release()

	tx, err := conn.BeginTx(ctxTO, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctxTO)

	if !regexp.MustCompile(`(?is)\bLIMIT\s+\d+`).MatchString(sql) {
		sql = fmt.Sprintf("WITH q AS (%s) SELECT * FROM q LIMIT %d", sql, limit)
	}
	rows, err := tx.Query(ctxTO, sql)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	flds := rows.FieldDescriptions()
	out := make([]map[string]any, 0, 16)
	for rows.Next() {
		vals, err := rows.Values()
		if err != nil {
			return nil, err
		}
		out = append(out, rowFromValues(flds, vals))
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctxTO); err != nil {
		return nil, err
	}
	return out, nil
}

// PaginatedResult holds pagination information
type PaginatedResult struct {
	Rows       []map[string]any
	Page       int
	PageSize   int
	TotalCount int
	HasMore    bool
	NextPage   int
}

func (s *Server) runPaginatedQuery(ctx context.Context, sql string, page, pageSize int) (*PaginatedResult, error) {
	// First, get total count
	countSQL := fmt.Sprintf("WITH query AS (%s) SELECT COUNT(*) FROM query", sql)

	ctxTO, cancel := context.WithTimeout(ctx, s.cfg.QueryTO)
	defer cancel()

	conn, err := s.db.Acquire(ctxTO)
	if err != nil {
		return nil, err
	}
	defer conn.Release()

	tx, err := conn.BeginTx(ctxTO, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctxTO)

	var totalCount int
	if err := tx.QueryRow(ctxTO, countSQL).Scan(&totalCount); err != nil {
		return nil, fmt.Errorf("failed to get total count: %w", err)
	}

	// Get paginated data
	offset := page * pageSize
	paginatedSQL := fmt.Sprintf("WITH query AS (%s) SELECT * FROM query LIMIT %d OFFSET %d", sql, pageSize, offset)

	rows, err := tx.Query(ctxTO, paginatedSQL)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	flds := rows.FieldDescriptions()
	out := make([]map[string]any, 0, pageSize)
	for rows.Next() {
		vals, err := rows.Values()
		if err != nil {
			return nil, err
		}
		out = append(out, rowFromValues(flds, vals))
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	if err := tx.Commit(ctxTO); err != nil {
		return nil, err
	}

	hasMore := offset+len(out) < totalCount
	nextPage := page + 1
	if !hasMore {
		nextPage = 0
	}

	return &PaginatedResult{
		Rows:       out,
		Page:       page,
		PageSize:   pageSize,
		TotalCount: totalCount,
		HasMore:    hasMore,
		NextPage:   nextPage,
	}, nil
}

func (s *Server) runStreamingQuery(ctx context.Context, sql string, maxPages, pageSize int) ([]streamPageOutput, int, error) {
	// First, get total count
	countSQL := fmt.Sprintf("WITH query AS (%s) SELECT COUNT(*) FROM query", sql)

	ctxTO, cancel := context.WithTimeout(ctx, s.cfg.QueryTO*time.Duration(maxPages))
	defer cancel()

	conn, err := s.db.Acquire(ctxTO)
	if err != nil {
		return nil, 0, err
	}
	defer conn.Release()

	tx, err := conn.BeginTx(ctxTO, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return nil, 0, err
	}
	defer tx.Rollback(ctxTO)

	var totalCount int
	if err := tx.QueryRow(ctxTO, countSQL).Scan(&totalCount); err != nil {
		return nil, 0, fmt.Errorf("failed to get total count: %w", err)
	}

	// Calculate actual pages to fetch
	totalPages := (totalCount + pageSize - 1) / pageSize
	pagesToFetch := minNonZero(maxPages, totalPages)

	var pages []streamPageOutput

	// Fetch pages
	for page := 0; page < pagesToFetch; page++ {
		offset := page * pageSize
		paginatedSQL := fmt.Sprintf("WITH query AS (%s) SELECT * FROM query LIMIT %d OFFSET %d", sql, pageSize, offset)

		rows, err := tx.Query(ctxTO, paginatedSQL)
		if err != nil {
			return nil, 0, err
		}

		flds := rows.FieldDescriptions()
		pageRows := make([]map[string]any, 0, pageSize)
		for rows.Next() {
			vals, err := rows.Values()
			if err != nil {
				rows.Close()
				return nil, 0, err
			}
			row := rowFromValues(flds, vals)
			pageRows = append(pageRows, row)
		}
		rows.Close()

		if err := rows.Err(); err != nil {
			return nil, 0, err
		}

		pages = append(pages, streamPageOutput{
			Page: page,
			Rows: pageRows,
		})

		// Stop if no more rows
		if len(pageRows) < pageSize {
			break
		}
	}

	if err := tx.Commit(ctxTO); err != nil {
		return nil, 0, err
	}

	return pages, totalCount, nil
}

func (s *Server) buildSearchSQL(ctx context.Context, q string, limit int) (string, error) {
	const meta = `
SELECT table_schema, table_name, column_name
FROM information_schema.columns
WHERE data_type IN ('text','character varying','character','citext')
  AND table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY table_schema, table_name, ordinal_position;
`
	ctxTO, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	rows, err := s.db.Query(ctxTO, meta)
	if err != nil {
		return "", err
	}
	defer rows.Close()

	type col struct{ s, t, c string }
	var cols []col
	for rows.Next() {
		var c col
		if err := rows.Scan(&c.s, &c.t, &c.c); err != nil {
			return "", err
		}
		cols = append(cols, c)
	}
	if len(cols) == 0 {
		return "", errors.New("no searchable columns")
	}
	like := strings.ReplaceAll(q, "'", "''")
	var parts []string
	for _, c := range cols {
		parts = append(parts, fmt.Sprintf(
			`SELECT '%s.%s' AS source_table, '%s' AS column, LEFT(CAST("%s" AS text), 240) AS match_text FROM "%s"."%s" WHERE "%s" ILIKE '%%%s%%'`,
			c.s, c.t, c.c, c.c, c.s, c.t, c.c, like,
		))
		if len(parts) >= 60 {
			break
		}
	}
	sql := "WITH u AS (\n" + strings.Join(parts, "\nUNION ALL\n") + fmt.Sprintf("\n) SELECT * FROM u LIMIT %d", limit)
	return sql, nil
}

func (s *Server) generateSQL(ctx context.Context, question, schema string, maxRows int) (string, string, error) {
	sys := s.sqlSystemPrompt(schema, maxRows)
	user := "Question: " + strings.TrimSpace(question) + `
Return ONLY SQL (or a single CANNOT_ANSWER line), nothing else.`

	raw, err := s.chat(ctx, sys, user, maxModelTokens)
	if err != nil {
		return "", "", err
	}
	return cleanSQL(raw), "model=" + s.model, nil
}

// sqlSystemPrompt is the shared instruction block for SQL generation and repair.
func (s *Server) sqlSystemPrompt(schema string, maxRows int) string {
	return `You translate plain English questions into a SINGLE, safe PostgreSQL query for ANY PostgreSQL database.

	Core Rules:
	- Use only read-only SQL (WITH/SELECT). No writes, DDL, or side effects.
	- Use proper JOINs based on foreign key relationships shown in the schema.
	- Always include an explicit LIMIT <= ` + fmt.Sprint(maxRows) + `.
	- Do not add semicolons.
	- Return concise, meaningful column aliases.
	- CRITICAL: Use table and column names EXACTLY as shown in the schema below, including quotes when present.

	Query Scope Rules:
	- SINGULAR questions ("Who is the...", "What is the...") -> LIMIT 1
	- PLURAL questions ("Who are the...", "What are the...") -> LIMIT 20
	- COUNT questions ("How many...") -> Return COUNT, no additional LIMIT
	- LIST questions ("List all...", "Show all...") -> LIMIT 50
	- COMPARISON questions ("Compare X and Y...") -> Return just the compared items
	
	User Override Rules (when user explicitly wants more results):
	- "Show ALL [items]" or "List ALL [items]" -> Use larger LIMIT (200-500)
	- "Give me EVERY [item]" -> Use larger LIMIT (200-500) 
	- "Show me EVERYTHING" -> Use larger LIMIT (200-500)
	- "Complete list of [items]" -> Use larger LIMIT (200-500)
	- When user emphasizes ALL/EVERY/COMPLETE -> Override normal limits
	- But still respect the maximum LIMIT constraint provided

	CRITICAL COLUMN CHECKING RULES:
	- BEFORE writing ANY SQL, verify EVERY column exists in the table you're using
	- ONLY use columns that are explicitly listed in the schema below
	- If you need a column that doesn't exist in your target table, you MUST use JOINs
	- Example: If you need user_id but you're querying order_items (which has no user_id), 
	  you MUST JOIN: order_items -> orders -> users via the foreign keys shown in schema
	- NEVER assume standard columns like 'id' exist - many tables use composite keys
	- For counting records: use COUNT(*) instead of COUNT(table.id) unless 'id' is explicitly shown
	- NEVER write SQL with non-existent columns - this will cause errors

	When the Question Cannot Be Answered:
	- If the question asks for a table or attribute that does not exist in the schema and cannot be reached through the FK lines, do NOT guess.
	- NEVER fabricate a requested attribute by aliasing an unrelated column to its name or by selecting NULL AS <name>.
	- Instead reply with exactly one line and no SQL:  CANNOT_ANSWER: <one or two sentences naming what is missing and the closest tables/columns that do exist>

	Universal Data Handling:
	- Work ONLY with tables and columns shown in the schema summary below
	- NEVER assume columns exist - only use columns explicitly listed in the schema
	- NEVER assume specific data values, enum values, or business logic
	- NEVER filter by assumed status values (completed, active, etc.) unless explicitly mentioned
	- If user asks for "top X" or "most Y", aggregate and sort the available data as-is
	- Use column names and relationships exactly as they appear in the schema
	- When in doubt, include more data rather than filtering it out
	- Focus on structural relationships (JOINs) rather than data content assumptions

	CRITICAL Identifier Rules (PostgreSQL Case Sensitivity):
	- PostgreSQL identifiers are case-sensitive when quoted with double quotes
	- Use table and column names EXACTLY as they appear in the schema below
	- If the schema shows "Book" (with quotes), you MUST use "Book" in your SQL
	- If the schema shows book (no quotes), you can use book, Book, or BOOK
	- NEVER change the case or remove quotes from identifiers shown in the schema
	- When in doubt, copy the identifier exactly as shown in the schema

	CRITICAL Counting and Aggregation Rules:
	- For counting records, use COUNT(*) or COUNT(1) instead of COUNT(table.id)
	- Only use COUNT(column_name) if that column is explicitly listed in the schema
	- Many tables use composite primary keys and don't have an 'id' column
	- For aggregating quantities or amounts, use SUM(column_name) where column_name exists
	- Always verify the column exists in the schema before using it in COUNT, SUM, AVG, etc.

	JOIN Strategy (CRITICAL - Generic approach for ANY database):
	- ALWAYS check the schema summary for foreign key relationships before writing JOINs
	- Look for "FK" lines in the schema that show: table1(column1) -> table2(column2)
	- If a column doesn't exist in the target table, trace the foreign key path in the schema
	- Example: If table A has column X, but you need column Y from table B, look for FK A.some_id -> B.id
	- Use ONLY the foreign key relationships explicitly shown in the schema summary
	- When multiple JOIN paths exist, choose the most direct one with fewest tables
	
	Performance Guidelines (CRITICAL):
	- PREFER single-table queries when possible
	- When JOINs are necessary, use ONLY the foreign key relationships explicitly shown in schema
	- Limit JOINs to maximum 2 tables to avoid expensive operations
	- Use INNER JOINs instead of LEFT JOINs when possible
	- If a question requires more than 2 JOINs, simplify to a single-table approximation
	- NEVER assume columns exist in the wrong table - always verify against schema first

	MANDATORY: Study this schema summary carefully before writing SQL. It shows all tables, columns, and foreign key relationships.
	Text in {braces} after a column, or after -- at the end of a TABLE line, is a human-written description of that element; use it to resolve meaning.
	
	` + schema + `
	
	REMEMBER: If you need a column that doesn't exist in your target table, find the FK relationship above and use JOINs.`
}

// cleanSQL strips the decoration models add around SQL: code fences, a "sql" language tag,
// and a trailing ';' (callers embed the query as a subquery, where ';' is a syntax error).
func cleanSQL(raw string) string {
	sql := strings.TrimSpace(raw)
	// Models sometimes wrap the query in prose with a fenced block in the middle; the fenced
	// part is the answer. Use the last fence in case an earlier one quotes the question.
	if m := fencedSQLRe.FindAllStringSubmatch(sql, -1); len(m) > 0 {
		sql = strings.TrimSpace(m[len(m)-1][1])
	}
	sql = strings.Trim(sql, "```")
	sql = strings.TrimSpace(strings.TrimPrefix(sql, "sql"))
	return strings.TrimRight(sql, "; \t\r\n")
}

var fencedSQLRe = regexp.MustCompile("(?s)```[a-zA-Z]*[ \t]*\n(.*?)```")

const cannotAnswerPrefix = "CANNOT_ANSWER:"

// classifyModelReply separates a SQL candidate from a reply that should not be executed:
// either the explicit CANNOT_ANSWER sentinel the prompt allows, or free-form prose (a model
// explaining itself despite instructions). Running prose as SQL only produces a meaningless
// syntax error, and a repair round on that error tends to yield placeholder queries like
// SELECT 1 that look like success. The returned reason is the model's text with the
// sentinel stripped.
func classifyModelReply(reply string) (sql string, declined bool, reason string) {
	trimmed := strings.TrimSpace(reply)
	if i := strings.Index(trimmed, cannotAnswerPrefix); i >= 0 {
		return "", true, strings.TrimSpace(trimmed[i+len(cannotAnswerPrefix):])
	}
	if !looksLikeSQL(trimmed) && !sqlKeywordRe.MatchString(trimmed) {
		return "", true, trimmed
	}
	return trimmed, false, ""
}

// sqlKeywordRe is the fallback SQL signal for text that does not start with a statement
// keyword (e.g. a misspelled SELECT): keep executing it so the database reports the real
// error and the repair round gets a chance. Prose that happens to contain these words takes
// the same path and ends in the repair round, which may itself decline.
var sqlKeywordRe = regexp.MustCompile(`(?i)\b(select|from|where|join|group by|order by|limit)\b|\bcount\(`)

// looksLikeSQL reports whether text starts like a read-only PostgreSQL statement.
func looksLikeSQL(text string) bool {
	first := strings.ToUpper(text)
	for strings.HasPrefix(first, "(") || strings.HasPrefix(first, "--") || strings.HasPrefix(first, "/*") {
		switch {
		case strings.HasPrefix(first, "("):
			first = strings.TrimSpace(first[1:])
		case strings.HasPrefix(first, "--"):
			nl := strings.IndexByte(first, '\n')
			if nl < 0 {
				return false
			}
			first = strings.TrimSpace(first[nl+1:])
		default:
			end := strings.Index(first, "*/")
			if end < 0 {
				return false
			}
			first = strings.TrimSpace(first[end+2:])
		}
	}
	for _, kw := range []string{"SELECT", "WITH", "VALUES", "TABLE", "EXPLAIN", "SHOW"} {
		if strings.HasPrefix(first, kw) && (len(first) == len(kw) || !isIdentByte(first[len(kw)])) {
			return true
		}
	}
	return false
}

func isIdentByte(c byte) bool {
	return c == '_' || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
}

// rowFromValues builds a JSON-friendly row. pgx hands UUIDs back as [16]byte, which encodes
// as a 16-element integer array; render them in canonical text form instead.
func rowFromValues(flds []pgconn.FieldDescription, vals []any) map[string]any {
	row := make(map[string]any, len(flds))
	for i, f := range flds {
		row[string(f.Name)] = normalizeValue(vals[i])
	}
	return row
}

func normalizeValue(v any) any {
	switch x := v.(type) {
	case [16]byte:
		return formatUUID(x)
	case [][16]byte:
		out := make([]string, len(x))
		for i := range x {
			out[i] = formatUUID(x[i])
		}
		return out
	case []any: // array columns arrive as []any with per-element Go values
		out := make([]any, len(x))
		for i := range x {
			out[i] = normalizeValue(x[i])
		}
		return out
	}
	return v
}

func formatUUID(b [16]byte) string {
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// chat sends one system+user exchange to the model with the configured sampling, timeout
// and thinking settings and returns the assistant text.
func (s *Server) chat(ctx context.Context, sys, user string, maxTokens int64) (string, error) {
	llmTO := s.cfg.LLMTO
	if llmTO <= 0 {
		llmTO = defaultLLMTimeout
	}
	ctxTO, cancel := context.WithTimeout(ctx, llmTO)
	defer cancel()

	var reqOpts []option.RequestOption
	if s.cfg.DisableThinking {
		// Not part of the OpenAI schema; vLLM and sglang forward it to the chat template.
		reqOpts = append(reqOpts, option.WithJSONSet("chat_template_kwargs.enable_thinking", false))
	}

	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(s.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(sys),
			openai.UserMessage(user),
		},
		MaxTokens: openai.Int(maxTokens),
	}
	if s.cfg.Temperature != nil {
		params.Temperature = openai.Float(*s.cfg.Temperature)
	}
	if s.cfg.TopP != nil {
		params.TopP = openai.Float(*s.cfg.TopP)
	}

	resp, err := s.llm.Chat.Completions.New(ctxTO, params, reqOpts...)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("model returned no choices")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// formatAttempts renders failed attempts for inclusion in a prompt.
func formatAttempts(attempts []sqlAttempt) string {
	var b strings.Builder
	for i, a := range attempts {
		fmt.Fprintf(&b, "Attempt %d SQL:\n%s\n", i+1, a.SQL)
		if a.SQLState != "" {
			fmt.Fprintf(&b, "Attempt %d error (SQLSTATE %s): %s\n\n", i+1, a.SQLState, a.Error)
		} else {
			fmt.Fprintf(&b, "Attempt %d error: %s\n\n", i+1, a.Error)
		}
	}
	return b.String()
}

// repairSQL asks the model for a corrected query given the exact database error(s) the
// previous attempt(s) produced. It reuses the generation prompt so all the schema rules apply.
func (s *Server) repairSQL(ctx context.Context, question, schema string, maxRows int, attempts []sqlAttempt) (string, error) {
	sys := s.sqlSystemPrompt(schema, maxRows)
	user := "Question: " + strings.TrimSpace(question) + `

The following SQL FAILED when executed against the database:

` + formatAttempts(attempts) + `Write a corrected query that fixes this error. Check every table and column you use against the schema; if a column is not listed for a table, find it in another table and JOIN via the FK lines.
If the question genuinely cannot be answered from this schema, reply with one CANNOT_ANSWER: line instead of inventing a placeholder query.
Return ONLY SQL (or a single CANNOT_ANSWER line), nothing else.`

	raw, err := s.chat(ctx, sys, user, maxModelTokens)
	if err != nil {
		return "", err
	}
	return cleanSQL(raw), nil
}

// explainFailure asks the model to say, in prose, why the attempts failed and what would be
// needed to answer the question, so the caller gets more than a bare database error.
func (s *Server) explainFailure(ctx context.Context, question, schema string, attempts []sqlAttempt) (string, error) {
	sys := `You are a PostgreSQL expert helping debug a failed translation of a plain-English question into SQL.
You will see the database schema, the question, and the SQL attempts that failed together with the exact database errors.
In at most four sentences, explain why the attempts failed and what is actually needed to answer the question:
name the concrete tables, columns and joins involved, or say what information is missing or ambiguous in the question.
Do not output SQL. Do not apologize.

Schema (text in {braces} after a column, or after -- at the end of a TABLE line, is a human-written description):

` + schema

	user := "Question: " + strings.TrimSpace(question) + "\n\n" + formatAttempts(attempts) + "Explain the failure."

	return s.chat(ctx, sys, user, maxExplainTokens)
}

// errModelDeclined marks an outcome where the model said the question cannot be answered
// from the schema; Explanation carries its reason.
var errModelDeclined = errors.New("the model could not translate the question into SQL for this schema")

// sqlOutcome is the result of generateAndRun: either SQL that executed, or the history of
// what was tried, the final error, and the model's explanation.
type sqlOutcome struct {
	SQL         string
	Note        string
	Attempts    []sqlAttempt // failed attempts, oldest first
	Explanation string
	Err         error // nil on success
}

func (o sqlOutcome) failureNote() string {
	switch {
	case o.Note == "":
		return "sql generation failed"
	case errors.Is(o.Err, errModelDeclined) && len(o.Attempts) == 0:
		return o.Note + " (model declined: question cannot be answered from this schema)"
	case errors.Is(o.Err, errModelDeclined):
		return fmt.Sprintf("%s (model declined after %d failed attempt)", o.Note, len(o.Attempts))
	case len(o.Attempts) == 0:
		return o.Note + " (query failed)"
	default:
		return fmt.Sprintf("%s (query failed after %d attempts)", o.Note, len(o.Attempts))
	}
}

// describeErr turns an execution error into the message and SQLSTATE shown to the model and
// the caller. Postgres errors carry detail and hints that are worth passing along verbatim.
func describeErr(err error) (msg, sqlstate string) {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		msg = pgErr.Message
		if pgErr.Detail != "" {
			msg += ". " + pgErr.Detail
		}
		if pgErr.Hint != "" {
			msg += ". Hint: " + pgErr.Hint
		}
		return msg, pgErr.Code
	}
	return err.Error(), ""
}

func newAttempt(sql string, err error) sqlAttempt {
	msg, state := describeErr(err)
	return sqlAttempt{SQL: sql, Error: msg, SQLState: state}
}

// generateAndRun drives up to three model rounds: generate SQL; if it fails, repair it using
// the database's own error; if that fails too, ask for an explanation instead of more SQL.
// run executes a candidate (after the read-only guard) and captures its own results.
func (s *Server) generateAndRun(ctx context.Context, question, schema string, maxRows int, run func(sql string) error) sqlOutcome {
	var out sqlOutcome

	guardAndRun := func(sql string) error {
		if err := guardReadOnly(sql); err != nil {
			return err
		}
		return run(sql)
	}

	reply, note, err := s.generateSQL(ctx, question, schema, maxRows)
	if err != nil {
		out.Err = fmt.Errorf("sql generation failed: %w", err)
		return out
	}
	out.Note = note
	sql, declined, reason := classifyModelReply(reply)
	if declined {
		out.Explanation = reason
		out.Err = errModelDeclined
		return out
	}
	out.SQL = sql

	runErr := guardAndRun(sql)
	if runErr == nil {
		return out
	}
	out.Attempts = append(out.Attempts, newAttempt(sql, runErr))
	out.Err = runErr
	if ctx.Err() != nil {
		return out
	}
	log.Debug().Str("sql", sql).Err(runErr).Msg("generated sql failed; attempting repair")

	// Round 2: repair using the database error.
	repairedReply, err := s.repairSQL(ctx, question, schema, maxRows, out.Attempts)
	if err != nil {
		log.Warn().Err(err).Msg("repair round failed to produce sql")
		return out
	}
	repaired, declined, reason := classifyModelReply(repairedReply)
	if declined {
		out.Explanation = reason
		out.Err = errModelDeclined
		return out
	}
	out.SQL = repaired
	runErr = guardAndRun(repaired)
	if runErr == nil {
		out.Err = nil
		out.Note += fmt.Sprintf(" (repaired after %d failed attempt)", len(out.Attempts))
		return out
	}
	out.Attempts = append(out.Attempts, newAttempt(repaired, runErr))
	out.Err = runErr
	if ctx.Err() != nil {
		return out
	}

	// Round 3: explain rather than guess again.
	explanation, err := s.explainFailure(ctx, question, schema, out.Attempts)
	if err != nil {
		log.Warn().Err(err).Msg("explanation round failed")
		return out
	}
	out.Explanation = explanation
	return out
}

func main() {
	// Handle version flag
	versionFlag := flag.Bool("version", false, "Print version information and exit")
	flag.Parse()

	if *versionFlag {
		fmt.Printf("pgmcp-server %s\n", version)
		fmt.Printf("  commit: %s\n", commit)
		fmt.Printf("  built:  %s\n", date)
		os.Exit(0)
	}

	zerolog.TimeFieldFormat = time.RFC3339
	switch strings.ToLower(envDefault("LOG_LEVEL", "info")) {
	case "debug":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	case "warn":
		zerolog.SetGlobalLevel(zerolog.WarnLevel)
	case "error":
		zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	default:
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	cfg := mustConfig()
	ctx := context.Background()

	srv, err := newServer(ctx, cfg)
	if err != nil {
		log.Fatal().Err(err).Msg("init failed")
	}
	impl := &mcp.Implementation{Name: "pgmcp-go", Version: "0.3.0"}

	server := mcp.NewServer(impl, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "ask",
		Description: "Answer questions about the connected PostgreSQL database by generating safe, read-only SQL. Automatically streams all results.",
	}, srv.handleAsk)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search free text across all tables/columns (ILIKE).",
	}, srv.handleSearch)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "stream",
		Description: "Stream large result sets by automatically fetching all pages. Returns complete results progressively.",
	}, srv.handleStream)

	// --- Streamable HTTP transport ---
	addr := envDefault("HTTP_ADDR", ":8080")
	path := envDefault("HTTP_PATH", "/mcp")
	bearer := strings.TrimSpace(os.Getenv("AUTH_BEARER"))

	base := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server { return server }, nil)
	var handler http.Handler = base
	if bearer != "" {
		handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			if strings.TrimSpace(got) != bearer {
				auditLog("auth_failed", r.RemoteAddr, "", "invalid bearer token", false)
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte("unauthorized"))
				return
			}
			base.ServeHTTP(w, r)
		})
	}

	// Apply middleware
	handler = requestSizeLimitMiddleware(handler)

	mux := http.NewServeMux()
	mux.Handle(path, handler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
		_, _ = w.Write([]byte("ok"))
	})

	// Create HTTP server
	httpServer := &http.Server{
		Addr:        addr,
		Handler:     mux,
		ReadTimeout: 30 * time.Second,
		// No WriteTimeout: the streamable HTTP transport keeps the GET SSE stream open
		// indefinitely, and tool calls are already bounded by LLMTO and QueryTO.
		IdleTimeout: 60 * time.Second,
	}
	srv.server = httpServer

	// Set up graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Info().Msg("received shutdown signal")

		// Give ongoing requests 30 seconds to complete
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Error().Err(err).Msg("error during shutdown")
		}
		os.Exit(0)
	}()

	log.Info().Str("addr", addr).Str("path", path).Msg("starting MCP server on streamable HTTP")
	auditLog("server_start", "system", "", addr, true)

	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal().Err(err).Msg("server error")
	}
}
