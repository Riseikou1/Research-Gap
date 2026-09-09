def sync_account(self, user_id: str, *, email: str | None, verified: bool) -> dict[str, object]:
    now = _now().isoformat()

    with self.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        connection.execute(
            """INSERT INTO accounts(user_id,email,email_verified,created_at,updated_at)
               VALUES(?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
               email=COALESCE(excluded.email,accounts.email),
               email_verified=MAX(accounts.email_verified,excluded.email_verified),
               updated_at=excluded.updated_at""",
            (user_id, email, int(verified), now, now),
        )

        if verified:
            connection.execute(
                """INSERT INTO credit_ledger
                   (entry_id,user_id,amount,kind,source,payment_reference,reason,created_at)
                   SELECT ?,?,?,'grant','free_lifetime',?,'Verified free-account allowance',?
                   WHERE NOT EXISTS (
                       SELECT 1
                       FROM credit_ledger
                       WHERE user_id=? AND source='free_lifetime'
                   )""",
                (
                    str(uuid4()),
                    user_id,
                    self.free_credits,
                    f"lifetime:{user_id}",
                    now,
                    user_id,
                ),
            )

        row = connection.execute(
            "SELECT * FROM accounts WHERE user_id=?",
            (user_id,),
        ).fetchone()

        connection.commit()

    return dict(row)