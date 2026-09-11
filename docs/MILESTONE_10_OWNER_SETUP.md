# Milestone 10 owner setup: account safety and report UX

Run the ordered database migrations before deploying the matching backend:

```bash
python -m src.persistence.migrate
```

Migration `0004_lifetime_credit_identity_registry.sql` creates a durable pseudonymous registry for
the one-time verified-account allowance. Migration `0005_provider_cache.sql` adds the durable,
versioned provider cache used when `DATABASE_URL` is configured. Existing verified accounts are
registered on their next authenticated backend request and their existing per-user unique grant
remains defense in depth.

## Render environment changes

Add `RESEARCH_GAP_LIFETIME_CREDIT_HMAC_SECRET` as a new, independent server-only secret. Generate it
with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`. Do not reuse the guest-cookie
secret. Back it up in the deployment secret store and do not rotate it without a deliberate registry
migration, because changing it changes future email identifiers. Keep the existing
`SUPABASE_SERVICE_ROLE_KEY`: account deletion now uses it to call the
Supabase Admin API. Deploy the migration before the application version and restart Render.

Set the server-only Vercel variable `RESEARCH_GAP_BACKEND_URL` to the public HTTPS origin of the
Render backend (for example, `https://research-gap-api.onrender.com`, with no `/analyses` suffix).
Remove the old `NEXT_PUBLIC_API_URL`; the browser now talks to the first-party `/api/backend` BFF.
Keep `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`. Never put the HMAC,
service-role key, OpenAI key, or Render backend secrets in a `NEXT_PUBLIC_*` value.

## Supabase checks

- Keep email confirmation enabled and the production Site URL/redirect URL accurate.
- Confirm the service-role credential on Render can delete Auth users through the Admin API.
- Keep the `avatars` bucket and server-role upload/delete access configured.
- Test deletion with an account that has no subscription, then confirm the identity disappears from
  Authentication → Users and its avatar object and saved analyses are gone.

## Deployment order and compatibility

Back up PostgreSQL, run migrations, deploy Render, set `RESEARCH_GAP_BACKEND_URL` on Vercel, then
deploy Vercel. Keep Render's allowed origin and app URL set to the exact Vercel/custom-domain HTTPS
origin and keep secure cookies enabled. Existing accounts keep their
balances, histories, roles, reservations, subscriptions, and user-ID lifetime-grant constraint.
Deleted accounts created before this migration whose email was already erased cannot be
retroactively entered into the new HMAC registry; monitor that legacy edge case. New deletions and
all existing active accounts are covered once they authenticate after deployment.

Before live billing, replace the $1/five-credit demo plan, validate unit economics and legal/tax
disclosures, create a new Stripe Price, configure the customer portal and production webhook, test
signed and replayed events, and only then set `RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING=true`.

## Scientific extraction smoke test

- With `full_text=false`, run the NAACL RAG example and confirm that its objective, enterprise
  workflow setting, RAG method, and abstract-supported findings appear. Do not expect named
  datasets, sample sizes, baselines, or metric names from that abstract.
- With `full_text=true`, confirm datasets, sample sizes, baselines, and named metrics only when the
  PDF was successfully accessed and each displayed claim has corresponding full-text evidence.
