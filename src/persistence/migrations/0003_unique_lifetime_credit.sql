-- A verified account receives one lifetime allowance, even under concurrent syncs.
CREATE UNIQUE INDEX credit_ledger_one_free_lifetime_per_user_idx
    ON credit_ledger (user_id)
    WHERE source = 'free_lifetime';
