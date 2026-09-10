# Milestone 9 owner setup and launch report

Last checked: 2026-09-08. This is an operational checklist for a non-expert owner, not legal, tax, accounting, or banking advice.

## What Codex completed

The repository contains a responsive Next.js web application, a private multi-principal FastAPI API, Supabase token verification and avatar-storage boundaries, a signed guest-session system, a transactional append-only credit ledger, server-owned Quick Search and Full Gap Analysis modes, durable job stages, protected exports, Stripe test Checkout/portal/webhooks, a server-stored administrator role and bootstrap command, an owner dashboard, ordered SQLite/PostgreSQL migrations, cleanup behavior, and deterministic tests. The scientific Milestones 1–8 pipeline and CLI remain shared with the API.

External accounts were not created. No administrator was created, no email was sent, no Stripe product exists yet, no bank account was connected, and no live payment or provider call was made. Those statements must remain true until you complete and verify the steps below.

## Local startup

Install and migrate the backend:

```bash
cd /path/to/research-gap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m src.persistence.migrate
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd /path/to/research-gap/web
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`; API health is `http://127.0.0.1:8000/health`. Run checks with:

```bash
source .venv/bin/activate
python -m pytest -q
cd web
npm test
npm run lint
npm run typecheck
npm run build
```

The browser calls FastAPI; it does not reproduce the research pipeline. Local mode without Supabase supports the historical trusted local API and explicit guest Quick Search, but real sign-up, Full Gap Analysis accounts, avatars, admin login, and billing require the integrations below.

## Environment and secrets checklist

Backend `.env` values:

| Name | Visibility | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | secret | Full-analysis model calls; never sent to the browser |
| `OPENAI_MODEL`, `OPENAI_EXTRACTION_MODEL`, `OPENAI_EMBEDDING_MODEL` | non-secret | Configured provider model names |
| `OPENALEX_API_KEY`, `OPENALEX_MAILTO` | secret/contact | OpenAlex access and polite-pool contact |
| `SUPABASE_URL` | public | Project URL |
| `SUPABASE_ANON_KEY` | public client key | May be used by the browser; RLS still matters |
| `SUPABASE_SERVICE_ROLE_KEY` | critical secret | Admin bootstrap and server avatar storage only |
| `SUPABASE_JWT_AUDIENCE`, `SUPABASE_JWT_ISSUER` | non-secret | Expected server-side token claims |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | critical secret | Use `sk_test_…` and `whsec_…` locally |
| `STRIPE_PRICE_ID` | non-secret identifier | Server-authoritative recurring Price ID |
| `RESEARCH_GAP_PAID_PRICE_DISPLAY_USD` | non-secret | Display-only amount; must match the configured Stripe Price |
| `RESEARCH_GAP_GUEST_COOKIE_SECRET` | critical secret | Generate with `python -c 'import secrets; print(secrets.token_urlsafe(48))'` |
| `RESEARCH_GAP_LIFETIME_CREDIT_HMAC_SECRET` | critical secret | Separate key for pseudonymous one-time-credit identities; never send to Vercel |
| `RESEARCH_GAP_APP_URL` | public | Frontend origin, without trailing slash |
| `RESEARCH_GAP_ALLOWED_ORIGINS` | public | Comma-separated exact origins; never `*` |
| `RESEARCH_GAP_SECURE_COOKIES` | non-secret | `false` for HTTP localhost, `true` behind production HTTPS |
| `RESEARCH_GAP_TRUSTED_LOCAL_MODE` | non-secret safety switch | Keep `false`; only the historical single-owner loopback API may set `true` |
| `RESEARCH_GAP_DATABASE_PATH` | private path | Durable SQLite file on persistent storage |
| `DATABASE_URL` | critical secret | Supabase Session Pooler PostgreSQL URL; takes precedence over the SQLite path |

Frontend `web/.env.local` contains only `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, and `NEXT_PUBLIC_SUPABASE_ANON_KEY`. Never add service-role, Stripe secret/webhook, OpenAI, or OpenAlex keys to `NEXT_PUBLIC_*` variables.

Edit the placeholder owner contact, About Me copy, and project link in `web/lib/content.ts` before publishing. That is the single site-content module; replace `owner@example.com`.

If a secret leaks: disable/roll it in the provider dashboard, update the deployment secret store, restart services, inspect access/webhook/admin logs, invalidate affected sessions, remove it from Git history if committed, and document the incident. Do not merely delete the current file.

## Supabase authentication, email, and avatars

1. Create a Supabase project at [Supabase](https://supabase.com/dashboard). In Project Settings → API, copy the Project URL, anon/publishable key, and service-role key to the fields above.
2. In Authentication → Providers → Email, enable email/password and require confirmation. Configure Site URL as `http://localhost:3000` for development and the HTTPS production URL later.
3. Add Redirect URLs `http://localhost:3000/profile` and `https://YOUR_DOMAIN/profile`. Configure password-reset and verification templates. For production, configure Authentication → Email/SMTP with an owner-controlled, domain-authenticated sender; verify SPF, DKIM, and DMARC and test delivery/reset links.
4. Create a public Storage bucket named `avatars`. Apply policies so public reads are allowed only if intended; uploads/deletes in this implementation use the server service role and paths are fixed to `{authenticated-user-id}/profile.{ext}`. Add a lifecycle/deletion policy for abandoned variants. Review [Supabase Storage access control](https://supabase.com/docs/guides/storage/security/access-control).
5. Set `SUPABASE_JWT_ISSUER` to `https://PROJECT_REF.supabase.co/auth/v1`. FastAPI verifies signature, issuer, audience, expiry, and subject through the project JWKS; it does not trust browser metadata for roles.
6. Start both apps, sign up, click the verification link, sign in, and confirm `/me` shows `verified: true` and exactly `2` credits. A refresh and another browser/device should show the same ledger balance.

Bootstrap the one administrator only after the Supabase server values and an owner-controlled email are configured:

```bash
python -m src.admin.bootstrap --email YOUR_OWNER_EMAIL
```

The command creates the auth user, generates a unique `owner-…` display name and high-entropy password, stores only the server-side role, and prints the password once. Put it in a password manager immediately and change it after first login. Running it again after success reports the existing admin and does not create another credential. Verify a normal user receives `403` from `/admin/summary`, then verify the administrator can open `/admin`. Never paste the password into this report, Git, screenshots, shell history, issue trackers, or logs.

## Stripe test mode

1. In the Stripe Dashboard, turn on **Test mode**. Create Product “Research GAP demo plan”; create a recurring monthly Price for **USD 1.00**. Copy its `price_…` identifier into `STRIPE_PRICE_ID`. Prices are immutable amounts: to change $1 later, create a new Price rather than editing the old amount.
2. Copy Developers → API keys → test secret key into `STRIPE_SECRET_KEY`. Do not expose it to Next.js. Configure Billing → Customer portal to allow subscription cancellation and payment-method updates.
3. Install/login to the Stripe CLI and forward signed events:

```bash
stripe login
stripe listen --forward-to http://127.0.0.1:8000/billing/webhook
```

Put the printed `whsec_…` in `STRIPE_WEBHOOK_SECRET`, then restart FastAPI. Production’s endpoint is `https://YOUR_API_DOMAIN/billing/webhook` and uses the endpoint-specific signing secret from Developers → Webhooks.

4. Use [Stripe test cards](https://docs.stripe.com/testing), including `4242 4242 4242 4242` for success and documented decline/authentication cards for failure. Use any future expiry/CVC only in test mode.
5. Confirm a Checkout browser return changes no balance. Confirm the signed `invoice.paid` event adds exactly `RESEARCH_GAP_PAID_CYCLE_CREDITS=5` once. In Developers → Events, resend the same event and confirm it is reported duplicate with no second grant. Test `invoice.payment_failed`, `customer.subscription.updated`, and `customer.subscription.deleted`; retry failed deliveries from Stripe after correcting configuration. Review [Stripe webhook signatures/retries](https://docs.stripe.com/webhooks).
6. Test a renewal with a Stripe test clock where available. Confirm cancellation stops future cycles but does not fabricate a refund or erase already granted credits. Incomplete/failed payments grant nothing.

The API never receives card data. Checkout and the customer portal are provider-hosted. The displayed “$1/month for 5 credits” is explicitly test/demo pricing and may lose money.

## Bank account and payouts

Never put a bank account number in this repository, database, `.env`, or support message. Enter payout details only in the official provider dashboard under Settings → Bank accounts and scheduling after the provider has verified the actual account owner, business, identity, address, tax information, and supported bank/currency.

As checked on 2026-09-08, [Stripe global availability](https://stripe.com/global) does not list South Korea or Mongolia as ordinary supported business-account countries, even though Stripe documents ways supported foreign businesses can accept [South Korean customer payment methods](https://docs.stripe.com/payments/countries/korea). Customer payment-method availability is not the same as eligibility to open a merchant account or receive local payouts. Do not infer eligibility from nationality or student residence. Confirm the real country of business, legal entity, tax residence, beneficial owner, and bank location directly with Stripe and qualified Korean/Mongolian/cross-border professionals. If direct onboarding is unavailable, compare provider-abstracted alternatives for that actual business location; do not silently route around onboarding or misstate an address/entity.

Before live activation, record payout currency, conversion costs, payout schedule, minimums, reserves, processing fees, refund/dispute/chargeback fees, statement descriptor, support contact, and reconciliation owner. For the first live charge: match Checkout → invoice → ledger grant → Stripe balance transaction. For the first payout: match balance transactions/fees/refunds to the bank deposit and accounting record; retain provider reports and investigate differences before scaling.

## Changing the plan and protecting cost

The authoritative allowance is `RESEARCH_GAP_PAID_CYCLE_CREDITS` in `.env` and parsed in `src/config.py`; webhook grants use that value in `src/api/routes/billing.py`. The authoritative charged amount is the Stripe Price referenced by `STRIPE_PRICE_ID`; `RESEARCH_GAP_PAID_PRICE_DISPLAY_USD` is display-only and must match it. The UI obtains both display-safe values from `/billing/plan`; do not duplicate a new business rule in React. Create/version a new Stripe Price, decide whether existing subscriptions remain grandfathered or are migrated with proration/notice, change the configured ID and display amount, test webhooks, and only then update public copy. The live-key startup guard requires `RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING=true`; setting it is an explicit owner action, not approval of the economics.

Inspect each result’s Technical details and admin aggregates for `stage_timings` and `work_metrics`. Provider token/cost fields are only available when upstream clients report them; absence must not be treated as zero cost. Set OpenAI project budgets/alerts and provider rate limits, cap application workers/candidates/evidence/full text, monitor cache hit rates, and add an application-wide monthly spend circuit breaker before live scale.

Measure both uncached/cached and abstract/full-text analyses. A basic unit-cost worksheet is:

```text
monthly price
- payment fixed and percentage fees
- (decomposition + query generation + embeddings + extraction + verification) provider cost × expected analyses
- OpenAlex/other data cost
- hosting + database + backups + storage + email + monitoring allocation
- tax/VAT/GST and invoicing cost
- expected refunds + disputes + abuse
- support/operations margin
= contribution margin
```

Do not sell five analyses for $1 until measurement shows a safe margin. Replace the placeholder and document the approval.

## Domain, deployment, and operations

Purchase the domain through an owner-controlled registrar account with MFA and renewal protection. Point frontend/API DNS records to chosen services, force HTTPS, set exact production URL/origin/issuer values, set secure cookies, and prefer a reverse proxy/same-origin `/api` deployment to simplify cookies/CORS. Production should set `DATABASE_URL` to the managed PostgreSQL Session Pooler URL; SQLite remains the local/test fallback and must not be placed on ephemeral production storage.

Back up the database and avatar bucket encrypted on a schedule. Before each release: stop writes or snapshot safely, back up, run `python -m src.persistence.migrate`, health-check, and retain a tested application rollback. SQL migrations are forward-only; database rollback means restoring the pre-migration backup after stopping the new service. Quarterly, perform a restore into an isolated environment and verify row counts, ownership, and ledger balance invariants.

Configure structured logs with secret/token/research-idea redaction, error tracking, uptime checks for `/health` and the frontend, alerts for provider/webhook/job failures, disk/backup monitoring, and retention cleanup. Schedule guest cleanup (startup already calls it) from a trusted process:

```bash
python -c 'from src.config import Settings; from src.persistence.database import Database; from src.persistence.repositories import AnalysisRepository; s=Settings.from_env(); d=Database(s.analysis_database_path, url=s.database_url); d.migrate(); print(AnalysisRepository(d).cleanup_expired_guests())'
```

Review login, export, Checkout, and global request throttling at the edge/WAF as well as application limits. Signed guest cookies plus pseudonymous network limiting reduce casual abuse; they cannot make guest abuse impossible.

## Legal, ownership, and research-data checklist

Original code/text/design generally receives copyright protection automatically when created, but registration procedure, evidence, enforcement benefits, work-for-hire/assignment, and cross-border effects depend on jurisdiction. Separately decide and document: repository software license; contributor/IP assignments; third-party dependency, font, icon, and dataset license/attribution audit; project/product trademark search and registration; domain registrant ownership; and business/company registration. Copyright, trademark, domain, company registration, and software licensing are different rights/processes.

Have qualified professionals review Terms of Service, Privacy Policy, Cookie Policy if non-essential cookies are added, acceptable-use rules, refund/cancellation policy, recurring-subscription renewal disclosures, contact/imprint requirements, research-data confidentiality and retention, breach response, and consumer withdrawal rights. Identify controller/processor roles and disclosures for OpenAI, OpenAlex, Supabase, Stripe, hosting, email, storage, monitoring, and any future analytics. Confirm cross-border transfers and data-processing agreements. Never call the analysis a systematic review or guaranteed novelty result.

Ask qualified Korean/Mongolian/business-location professionals about tax residency, business registration, VAT/GST/sales tax, invoicing, bookkeeping, foreign-exchange reporting, consumer law, privacy law, and subscription renewals. Check current official sources for the actual jurisdiction; do not rely on this document as a compliance guarantee. No included legal page has been reviewed by a lawyer.

## Pre-launch gate

- Replace test pricing and verify unit economics, refunds, cancellation, tax, and disclosures.
- Verify actual provider/business/bank eligibility and complete identity/business checks.
- Rotate all development secrets; use production secret storage and MFA.
- Verify no `.env`, database, token, service key, webhook secret, or generated admin credential is committed.
- Run all Python/frontend/build tests and a provider-test end-to-end pass.
- Test cross-user read/list/delete/claim/export denial, concurrent last-credit spending, failed-job refund, duplicate/out-of-order webhooks, suspension, and admin role enforcement.
- Test backup/restore and migration rollback; schedule guest/data retention cleanup.
- Test keyboard/screen reader, reduced motion, contrast, mobile/tablet/desktop layouts, long ideas/titles, empty states, 404/401/402/403/429/5xx, refresh polling, and export.
- Run dependency/security scans, review third-party licenses, and commission a security review/penetration test.
- Remove test banners only after live configuration and price acknowledgement are intentional and documented.
- Reconcile the first live payment and first payout before increasing limits.

## Account deletion and retained audit data

The API blocks deletion during active analyses or while a paid subscription remains active. It then deletes the Supabase Auth identity through the server-only Admin API, attempts avatar deletion, removes analyses, and anonymizes the local account. A keyed HMAC marker remains so the same normalized verified email cannot receive the lifetime allowance twice; the registry contains no plaintext deleted email. Ledger/payment/audit rows may retain a provider/user reference needed for financial/security reconciliation; define a lawful retention period before launch. Guest sessions expire after `RESEARCH_GAP_GUEST_RETENTION_HOURS` (default 72) and their analyses are removed by cleanup. Pre-Milestone-9 analyses have no owner and are intentionally unreadable through public APIs.
