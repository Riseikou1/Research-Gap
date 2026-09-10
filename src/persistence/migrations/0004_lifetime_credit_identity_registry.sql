-- Keep only a keyed, pseudonymous identity after deletion so a verified email
-- cannot receive the one-time allowance again through a new auth identity.
CREATE TABLE lifetime_credit_identities (
    identity_hmac TEXT PRIMARY KEY,
    first_user_id TEXT NOT NULL,
    granted_at TEXT NOT NULL
);

CREATE INDEX lifetime_credit_identity_user_idx
    ON lifetime_credit_identities (first_user_id);
