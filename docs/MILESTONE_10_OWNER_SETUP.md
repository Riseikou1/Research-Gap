# Milestone 10 owner setup and production runbook

This file lists actions that require the repository owner or an external dashboard. Nothing in the repository deploys services, creates vendor accounts, or stores production secrets.

## Local verification

Use Python 3.11 or 3.12 and Node 20. The exact CI sequence is:

```bash
python -m pip install -r requirements.lock
python -m pytest
python -m compileall -q main.py src tests
python -m src.persistence.migrate --database /tmp/research-gap-m10.sqlite3
cd web
npm ci
npm test
npm run lint
npm run typecheck
npm run build
cd ..
docker build -t research-gap:m10 .
git diff --check
```

Tests are offline and must not use paid provider calls.

## Secret ownership

Keep `OPENAI_API_KEY`, `OPENALEX_API_KEY`, `DATABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, the guest-cookie secret, and the lifetime-credit HMAC secret in the backend host only. Never add them to Vercel variables prefixed with `NEXT_PUBLIC_`, client bundles, Docker build arguments, health responses, or logs. The frontend receives only the Supabase public URL and anon key. Rotate a secret immediately if it appears in an image, log, support ticket, or build output.

Production must use unique random cookie/HMAC secrets, HTTPS secure cookies, exact allowed origins, a durable PostgreSQL `DATABASE_URL`, and `RESEARCH_GAP_TRUSTED_LOCAL_MODE=false`.

## Native Render backend

1. Create a PostgreSQL database and a Python web service from this repository.
2. Build with `python -m pip install -r requirements.lock`.
3. Run migrations as the pre-deploy command: `python -m src.persistence.migrate`.
4. Start with `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`.
5. Configure `/health/live` for process liveness and `/health/ready` for readiness.
6. Set `RESEARCH_GAP_BUILD_VERSION` to the release/commit identifier.
7. Add the required server-only variables above and set `RESEARCH_GAP_APP_URL` plus exact `RESEARCH_GAP_ALLOWED_ORIGINS`. Stripe secrets are required only if billing is enabled.
8. Set `RESEARCH_GAP_AUTO_MIGRATE=false`; production workers must not race migrations during startup.

Migrations are deliberately separate from process startup so multiple instances cannot all treat schema work as application boot work. The migration command reads only `DATABASE_URL` and `RESEARCH_GAP_DATABASE_PATH`, so missing Stripe/provider configuration does not block a release migration.

## Optional Docker backend

Build with `docker build -t research-gap:m10 .`. Run migrations once as a release job using the same image:

```bash
docker run --rm --env-file /secure/backend.env research-gap:m10 python -m src.persistence.migrate
docker run --rm -p 8000:8000 --env-file /secure/backend.env research-gap:m10
```

The image runs as an unprivileged user, contains no repository `.env`, database, cache, tests, or frontend dependencies, and uses liveness health checking. Do not bake secrets into the image.

## Vercel frontend

Set `RESEARCH_GAP_BACKEND_URL` as a server-only variable. Set only `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` as browser-visible values. Point the backend allowed origin and app URL at the final HTTPS Vercel domain. Verify CSP and same-origin proxy behavior after deployment.

## Supabase, Stripe, email, and domain actions

Billing is disabled by default with `RESEARCH_GAP_BILLING_ENABLED=false`. In this mode, do not set
dummy Stripe credentials: the application starts without them, paid purchase controls are disabled,
and billing endpoints return HTTP 503. Free/lifetime credits, analysis refunds and history, admin
access, and the deletion block for accounts with a recorded active subscription remain in force.
To enable billing, set `RESEARCH_GAP_BILLING_ENABLED=true` and provide all of
`STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, and `STRIPE_PRICE_ID`; startup fails if any is absent.
Live Stripe keys retain the additional `RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING=true` safeguard.

- Create the Supabase project, configure the production site URL and exact redirect URLs, enable email verification, and set JWT issuer/audience values on the backend.
- Keep the Supabase service-role key backend-only. Create the private avatar bucket and its owner-scoped policies before enabling uploads.
- Configure an email provider and sender domain; test verification and recovery delivery.
- Create Stripe products/prices in test mode first. Set the signed webhook endpoint to `/billing/webhook` and subscribe only to the documented invoice/subscription events. Never put bank or payout details in repository files.
- Validate the custom domain, DNS, TLS, cookie domain, CORS origin, Supabase redirects, and Stripe return URLs together.
- Promote an administrator with the server-side bootstrap command documented in `ADMIN_CREDIT_EXEMPTION.md`; browser metadata cannot grant this role.

## Provider spending and pricing

The daily provider budget is off when `RESEARCH_GAP_DAILY_PROVIDER_BUDGET_USD=0`. To enable it, choose a conservative per-analysis reservation and explicitly enter current model pricing in `RESEARCH_GAP_OPENAI_INPUT_PER_MILLION_USD` and `RESEARCH_GAP_OPENAI_OUTPUT_PER_MILLION_USD`. Pricing is intentionally operator-managed because providers can change it. Confirm current prices in the provider dashboard before every release. Budget reservations occur before queueing and include administrators; application credit exemption does not exempt real provider cost.

Usage is recorded per provider/model/stage/analysis when the SDK exposes token metadata. Missing token metadata remains unavailable. Inspect `/admin/diagnostics` for the daily aggregate, budget commitment, worker state, and recent safe failure categories.

OpenAlex GET requests use bounded exponential backoff with jitter and honor numeric or HTTP-date `Retry-After`. Embedding calls may use the SDK's bounded retry because they are idempotent. Billable structured-generation calls disable implicit retries, and Stripe webhook processing remains signature-checked and idempotent rather than client-retried.

## Incident response and rollback

For provider-cost spikes, set the daily budget below current commitment or disable new traffic at the platform edge; do not delete accounting rows. For provider degradation, inspect request IDs and safe categories, then reduce worker concurrency if needed. For database incidents, fail readiness while retaining liveness, stop new jobs, restore PostgreSQL, run migrations once, and verify private history ownership before reopening.

Rollback application images independently from the database. Migrations are additive; do not remove columns/tables during an emergency rollback. Preserve analysis JSON, provider usage, credit ledger, budget reservations, audit entries, and webhook records. Rotate any possibly exposed secret and invalidate affected sessions.

## Pre-launch checklist

- All CI commands and the container build pass from a clean checkout.
- Liveness succeeds without a database; readiness fails when the database is unavailable.
- No secret is present in the frontend build, image history, logs, health output, or error response.
- Quick Search performs no OpenAI work and creates no citation graph.
- Full Analysis persists citation context, and an old graph-less result still renders.
- Admin jobs remain subject to rate, concurrency, paper, evidence, full-text, and provider-budget limits.
- Credit reservation/refund and budget reservation/release behavior are independently verified.
- Billing-disabled deployments show no purchase path and return safe 503 responses; billing-enabled deployments have all three Stripe variables and a signed webhook smoke test.
- The NAACL smoke check preserves exact evidence/provenance rules.
- Privacy/terms/contact pages and retention claims match actual configuration.
