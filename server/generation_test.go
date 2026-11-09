package main

import (
	"testing"
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
