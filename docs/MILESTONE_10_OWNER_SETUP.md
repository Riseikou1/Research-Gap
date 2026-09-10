# Milestone 10 owner setup: account safety and report UX

Run the ordered database migrations before deploying the matching backend:

```bash
python -m src.persistence.migrate
```

Migration `0004_lifetime_credit_identity_registry.sql` creates a durable pseudonymous registry for
the one-time verified-account allowance. Existing verified accounts are registered on their next
authenticated backend request and their existing per-user unique grant remains defense in depth.

## Render environment changes

Add `RESEARCH_GAP_LIFETIME_CREDIT_HMAC_SECRET` as a new, independent server-only secret. Generate it
with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`. Do not reuse the guest-cookie
secret. Back it up in the deployment secret store and do not rotate it without a deliberate registry
migration, because changing it changes future email identifiers. Keep the existing
`SUPABASE_SERVICE_ROLE_KEY`: account deletion now uses it to call the
Supabase Admin API. Deploy the migration before the application version and restart Render.

No new Vercel environment variable is required. Vercel continues to receive only
`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, and `NEXT_PUBLIC_SUPABASE_ANON_KEY`; never put the
HMAC or service-role key in a `NEXT_PUBLIC_*` value.

## Supabase checks

- Keep email confirmation enabled and the production Site URL/redirect URL accurate.
- Confirm the service-role credential on Render can delete Auth users through the Admin API.
- Keep the `avatars` bucket and server-role upload/delete access configured.
- Test deletion with an account that has no subscription, then confirm the identity disappears from
  Authentication → Users and its avatar object and saved analyses are gone.

## Deployment order and compatibility

Back up PostgreSQL, run migrations, deploy Render, then deploy Vercel. Existing accounts keep their
balances, histories, roles, reservations, subscriptions, and user-ID lifetime-grant constraint.
Deleted accounts created before this migration whose email was already erased cannot be
retroactively entered into the new HMAC registry; monitor that legacy edge case. New deletions and
all existing active accounts are covered once they authenticate after deployment.

Before live billing, replace the $1/five-credit demo plan, validate unit economics and legal/tax
disclosures, create a new Stripe Price, configure the customer portal and production webhook, test
signed and replayed events, and only then set `RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING=true`.
