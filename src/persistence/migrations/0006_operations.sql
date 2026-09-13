CREATE TABLE provider_usage (
    usage_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL,
    occurred_at TEXT NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX provider_usage_analysis_idx ON provider_usage(analysis_id, occurred_at);
CREATE INDEX provider_usage_time_idx ON provider_usage(occurred_at);

CREATE TABLE provider_budget_reservations (
    reservation_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL UNIQUE,
    budget_day TEXT NOT NULL,
    reserved_cost_usd REAL NOT NULL,
    actual_cost_usd REAL,
    status TEXT NOT NULL CHECK(status IN ('reserved','settled','released')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX provider_budget_day_idx ON provider_budget_reservations(budget_day, status);

CREATE TABLE operational_failures (
    failure_id TEXT PRIMARY KEY,
    request_id TEXT,
    analysis_id TEXT,
    stage TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX operational_failures_time_idx ON operational_failures(created_at);
