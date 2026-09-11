CREATE TABLE IF NOT EXISTS provider_cache (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    stored_at DOUBLE PRECISION NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (namespace, cache_key)
);
