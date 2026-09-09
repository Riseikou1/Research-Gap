-- Milestone 9 keeps pre-migration analyses private by leaving owner columns NULL.
ALTER TABLE analyses ADD COLUMN owner_kind TEXT CHECK (owner_kind IN ('user', 'guest', 'local'));
ALTER TABLE analyses ADD COLUMN owner_id TEXT;
ALTER TABLE analyses ADD COLUMN mode TEXT NOT NULL DEFAULT 'full' CHECK (mode IN ('quick', 'full'));
ALTER TABLE analyses ADD COLUMN stage TEXT NOT NULL DEFAULT 'preparing';
ALTER TABLE analyses ADD COLUMN progress_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE analyses ADD COLUMN reservation_id TEXT;

CREATE INDEX analyses_owner_created_idx ON analyses (owner_kind, owner_id, created_at DESC);

CREATE TABLE accounts (
    user_id TEXT PRIMARY KEY,
    email TEXT,
    display_name TEXT NOT NULL DEFAULT '',
    avatar_url TEXT,
    email_verified INTEGER NOT NULL DEFAULT 0 CHECK (email_verified IN (0, 1)),
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deleted')),
    stripe_customer_id TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE credit_ledger (
    entry_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES accounts(user_id),
    amount INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('grant', 'reservation', 'settlement', 'refund', 'admin_adjustment')),
    source TEXT NOT NULL,
    analysis_id TEXT,
    payment_reference TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(source, analysis_id),
    UNIQUE(source, payment_reference)
);
CREATE INDEX credit_ledger_user_idx ON credit_ledger (user_id, created_at);

CREATE TABLE credit_reservations (
    reservation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES accounts(user_id),
    analysis_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('reserved', 'settled', 'released')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE guest_sessions (
    guest_id TEXT PRIMARY KEY,
    network_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    claimed_by_user_id TEXT REFERENCES accounts(user_id)
);
CREATE INDEX guest_network_idx ON guest_sessions (network_key, created_at);

CREATE TABLE rate_events (
    event_id TEXT PRIMARY KEY,
    principal_key TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX rate_events_window_idx ON rate_events (principal_key, action, created_at);

CREATE TABLE subscriptions (
    user_id TEXT PRIMARY KEY REFERENCES accounts(user_id),
    stripe_customer_id TEXT UNIQUE,
    stripe_subscription_id TEXT UNIQUE,
    status TEXT NOT NULL DEFAULT 'none',
    current_period_end TEXT,
    last_event_created INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE payment_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    created_epoch INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processed', 'ignored', 'failed')),
    error_message TEXT,
    processed_at TEXT NOT NULL
);

CREATE TABLE admin_audit_log (
    audit_id TEXT PRIMARY KEY,
    admin_user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_user_id TEXT,
    reason TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX admin_audit_created_idx ON admin_audit_log (created_at DESC);
