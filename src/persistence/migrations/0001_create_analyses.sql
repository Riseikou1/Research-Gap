CREATE TABLE analyses (
    analysis_id TEXT PRIMARY KEY,
    research_idea TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    decomposer TEXT NOT NULL CHECK (decomposer IN ('deterministic', 'openai')),
    query_generator TEXT NOT NULL CHECK (query_generator IN ('deterministic', 'openai')),
    paper_limit INTEGER NOT NULL CHECK (paper_limit BETWEEN 1 AND 100),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    configuration_json TEXT NOT NULL,
    result_json TEXT,
    error_message TEXT,
    CHECK ((status = 'completed' AND result_json IS NOT NULL AND error_message IS NULL)
        OR (status = 'failed' AND result_json IS NULL AND error_message IS NOT NULL)
        OR (status IN ('pending', 'running') AND result_json IS NULL AND error_message IS NULL))
);

CREATE INDEX analyses_created_at_idx ON analyses (created_at DESC);
CREATE INDEX analyses_status_idx ON analyses (status);
