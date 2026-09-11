# Administrator credit exemption

Administrator Full Gap Analyses consume zero internal application credits. This is an operational
exception for a private administrator account, not a public subscription plan and not an infinite
ledger balance.

The backend grants the exemption only when the current `accounts.role` value in PostgreSQL is
`admin` and the account is active. Supabase user metadata, JWT custom metadata, email addresses,
display names, frontend state, and analysis request fields are not authorization sources. A role
change therefore applies on the next analysis-creation request.

For an exempt Full Gap Analysis, `configuration_json.credit_billing_decision` is stored as
`administrator_credit_exempt` and `reservation_id` remains null. No credit reservation or ledger
entry is created. Success, failure, and queued-job cancellation skip settlement/refund processing
because there is no reservation. The administrator's real stored balance remains unchanged.

This exemption does not make provider usage free. OpenAI extraction and embedding calls, OpenAlex
traffic, and full-text downloads can still incur financial or operational cost. Keep the
administrator account private and protected, and retain all configured concurrency limits, rolling
rate limits, project budgets, spending circuit breakers, candidate/evidence/full-text bounds,
input validation, scientific verification, ownership checks, and logging.

Ordinary verified users still reserve one internal credit before a Full Gap Analysis is queued.
Successful work settles that reservation; failed or cancelled work returns it using the existing
idempotent refund path. Guest and signed-in Quick Search behavior is unchanged.

After deployment, verify with a private administrator account that `/me` reports `credit_exempt`
as true, a zero-credit Full Gap Analysis is created with no reservation, the result remains owned
and visible in History, and the balance and ledger remain unchanged after both successful and
failed test jobs. Use deterministic/stubbed checks where possible: exemption from internal credits
does not exempt a live smoke test from real provider cost.
