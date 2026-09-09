# Codex Prompt — Milestone 9: Multi-user Web Interface, Usage Limits, Admin, and Payments

You are implementing Milestone 9 of the existing **Research GAP** repository. This is a large, security-sensitive milestone. Work directly in the current repository and finish the implementation; do not merely propose an architecture or return example snippets.

## Start here

Before changing anything:

1. Read `IMPORTANT.md` completely. It is the project contract.
2. Read `README.md` completely.
3. Inspect the current Git status and preserve all unrelated user changes.
4. Inspect the full Milestone 8 implementation, especially:
   - `src/api/`
   - `src/application/`
   - `src/persistence/`
   - `src/pipeline.py`
   - `src/config.py`
   - `tests/unit/test_milestone8_*.py`
5. Inspect the actual serialized pipeline result before designing frontend types or result components. Do not invent a second response schema.
6. Run the existing test suite before implementation and record the baseline. Do not make paid OpenAI, OpenAlex, authentication, email, or payment calls from tests.

Milestone 8 is the backend boundary. Preserve the existing scientific pipeline and CLI behavior. The web application must call the FastAPI service rather than reimplementing retrieval, ranking, extraction, landscape analysis, or gap verification in JavaScript.

The repository contains a file named `milestone-9-full-text-ingestion-prompt.md`, but full-text ingestion is already implemented and `IMPORTANT.md` defines Milestone 9 as the web interface. Treat the web application described here as Milestone 9. Update documentation so the numbering is unambiguous without deleting useful historical material.

## Required outcome

Build a polished, responsive Research GAP web application with:

- A public home page containing a brief product introduction, a clear limitations statement, and an About Me section.
- A separate analysis/search page. Do not place the actual search engine on the home page.
- Optional email-based sign-up/sign-in. The product must remain usable in a deliberately limited guest mode.
- User profiles with display name and profile picture.
- Strict private ownership of analyses and histories.
- A cheap guest preview, exactly two free full-analysis credits for each verified free account, and a functionally working payment flow in provider test mode.
- A placeholder paid plan of **USD $1 per month for 5 full-analysis credits per billing cycle**. This is a test placeholder, not an approved live commercial price.
- A secure administrator account and server-enforced administrator dashboard.
- Summary-first, evidence-rich results rather than a raw dump of CLI output.
- A detailed owner setup/report file explaining every action the repository owner must complete manually.
- Automated backend and frontend tests, documentation, migrations, and a complete final handoff.

In billing and UI language, one **analysis credit** means one user-submitted full analysis job. It does **not** mean every internal deterministic query, LLM-generated query, OpenAlex route, verification query, or model request made by the pipeline. Avoid the ambiguous label “query credit.”

## Technology and repository structure

Use the existing FastAPI application as the backend. Create a separate frontend application under `web/` using:

- Next.js with the App Router
- TypeScript in strict mode
- Tailwind CSS
- shadcn/ui or similarly accessible, locally owned components
- A maintained schema-validation library for external/API data
- A lightweight query/data-fetching solution only if it materially simplifies polling, caching, and error handling

Do not replace the Python pipeline with a full-stack JavaScript implementation. Do not introduce microservices, GraphQL, Redis, Docker, a vector database, or Kubernetes for this milestone.

Use a managed authentication system rather than implementing password security from scratch. Prefer **Supabase Auth** for verified email/password or email magic-link authentication and Supabase Storage for profile pictures, unless the repository already has a better established managed provider. Isolate authentication behind a small backend interface so it can be replaced. Verify user tokens server-side in FastAPI; hiding frontend controls is not authorization.

Continue to use the existing SQLite analysis persistence for this milestone unless a change is genuinely required. Add ordered migrations rather than rewriting the original schema. Keep the data-access layer replaceable because a later deployment milestone may move persistence to PostgreSQL.

Use **Stripe Checkout and Stripe Billing in test mode** for the placeholder recurring plan. Keep payment-provider code behind a narrow interface. The product must be testable locally with Stripe test keys and Stripe CLI/webhook forwarding. Never enable live payments automatically.

If an external account, secret, DNS record, verified email sender, payment product, webhook, or deployment service is unavailable, implement and test the integration boundary with fakes, document the exact missing owner action, and clearly report the blocker. Do not fabricate a successful external setup.

## Product access model

Implement these rules exactly unless an existing constraint makes one impossible; if so, explain it before choosing the safest equivalent.

### Guest

- A guest may use a **Quick Search** only.
- Quick Search performs bounded retrieval and inexpensive deterministic/basic ranking only. It must not invoke paid OpenAI decomposition, embeddings, evidence extraction, gap generation, verification, or full-text extraction.
- Limit guests to one Quick Search in a rolling 24-hour window using both a signed, HttpOnly guest-session cookie and a privacy-conscious server-side network rate-limit key. Do not claim this makes abuse impossible.
- Guest results are private to the guest session and must never appear in a global history.
- Guest records should have a documented short retention period and an automated or callable cleanup mechanism.
- A guest may sign up and claim their own still-valid guest analyses, but never anybody else’s.

### Verified free account

- Email verification is required before spending a paid OpenAI analysis credit.
- Each free account receives exactly **2 lifetime Full Gap Analysis credits**.
- The allowance must be enforced atomically on the server. It must survive refreshes, logouts, different devices, and concurrent requests.
- Do not reset free credits monthly.
- Quick Search can remain available with a conservative daily rate limit, configurable through settings.

### Placeholder paid plan

- Stripe test-mode recurring subscription: **$1/month**.
- Grant **5 Full Gap Analysis credits per successful billing cycle**.
- Make both the price reference and allowance configurable. Use a Stripe Price ID supplied through environment configuration rather than trusting a client-submitted amount.
- Credits are granted only after a verified, idempotently processed Stripe webhook reports a successful payment/subscription event.
- Do not grant access merely because the browser returned from Checkout successfully.
- Handle subscription creation, renewal, cancellation, incomplete payment, failed payment, and duplicate/out-of-order webhook delivery conservatively.
- Provide a Stripe customer portal link for signed-in subscribers where supported.
- Label the plan as test/demo pricing wherever it appears until the owner explicitly enables live mode.
- Add a production safety guard so placeholder pricing cannot accidentally go live without an explicit acknowledgement setting.

### Full-text cost

- Full-text analysis is available only to signed-in users with sufficient credit.
- For the initial implementation, one full analysis consumes one credit whether full text is enabled or not, but collect enough usage metadata to let the owner change this policy later.
- Clearly warn in the owner report that $1 for five analyses may be loss-making and must not be used as a real price until actual per-analysis cost is measured.

## Atomic credit accounting

Implement a proper append-only usage/credit ledger or an equally auditable transactional design. A mutable integer alone is insufficient.

Required behavior:

- Reserve a credit atomically before a full analysis is queued.
- Prevent double-spending under concurrent requests.
- Attach the reservation to the analysis ID.
- Settle the reservation when the analysis completes.
- Automatically release/refund the credit when the job fails before producing a usable result.
- Make retry and webhook operations idempotent.
- Record grant source, amount, analysis/payment reference, timestamp, and administrator adjustments.
- Never allow the browser to choose its own role, balance, product price, allowance, owner ID, or Stripe customer ID.
- Provide safe administrator credit adjustment with an audit reason.

## Authentication, sessions, and ownership

Extend the API and database so every record is scoped to exactly one authorized principal: a signed-in user or a guest session.

Requirements:

- Verify managed-auth access tokens in FastAPI using the provider’s supported server-side verification/JWKS mechanism.
- Use secure cookie settings appropriate to local development and production.
- Define an explicit CORS policy; never use permissive origins with credentials.
- Protect state-changing cookie-authenticated operations against CSRF as appropriate to the chosen flow.
- Do not accept `user_id` or owner identifiers from ordinary request bodies.
- `GET /analyses`, `GET /analyses/{id}`, and deletion must only expose the caller’s records.
- Knowing an analysis UUID must not grant access.
- Existing pre-migration local records must not become publicly readable. Define and test a safe legacy-record policy.
- Account deletion must define what happens to analyses, avatars, usage records, and the minimal payment/audit records that may need legal retention.
- Add explicit, bounded avatar file type and size validation. Do not trust a filename or browser MIME value alone.
- Do not store raw access tokens, passwords, API keys, Stripe secrets, or Supabase service-role secrets in the application database.

## Secure administrator account

The owner needs a distinct administrator login with a unique name and password.

Implement it securely:

1. Administrator privilege must be stored and enforced server-side. It must never be assignable through public sign-up, profile editing, client metadata, or a normal API request.
2. Add a one-time, idempotent administrator bootstrap command/script that creates the admin through the selected auth provider and records the server-side admin role.
3. Generate a unique administrator username/display name and a cryptographically random high-entropy password. The administrator must still have a valid owner-controlled login email because normal authentication is email-based.
4. Never hard-code the username, password, email, password hash, service token, or recovery secret in committed source, fixtures, documentation, migrations, screenshots, or frontend bundles.
5. Never write the plaintext password to `OWNER_SETUP.md` or any tracked file.
6. Print the generated password once during successful bootstrap and instruct the owner to store it in a password manager and change it after first login. Avoid logging it anywhere else.
7. If the external authentication account cannot actually be created because owner configuration is missing, do not pretend it exists. Finish the code, report exactly what is missing, and provide the single bootstrap command to run after configuration.
8. In the final response, explicitly tell the owner whether the administrator was actually created. If it was created, provide the generated admin username and one-time password as requested and warn that they were not committed. If it was not created, say so plainly and provide the exact creation command. Do not claim credentials work unless verified.

Create a protected `/admin` dashboard with server-side role enforcement. It should show operational information useful to one owner:

- User/account status and plan, with minimal necessary personal data
- Analysis counts, stages, failures, and aggregate model usage/cost data
- Guest/free/paid usage summaries
- Subscription status synchronized from Stripe
- Safe user suspension/reactivation if implemented
- Credit adjustment with required reason
- Failed webhook visibility and retry guidance
- Audit log of administrator actions

Do not display user passwords, authentication tokens, payment card data, full Stripe secrets, or provider API keys. By default, avoid exposing private unpublished research text in broad admin tables; require a deliberate detail action and explain the privacy implication.

## Analysis API modes and job progress

Preserve the existing complete analysis behavior but expose an explicit server-owned mode:

- `quick`: retrieval/basic ranking only, no paid OpenAI path
- `full`: evidence, literature landscape, gap candidates, direct assessment, verification, and optional full text

Do not make frontend flags the source of truth for billing or pipeline selection.

The existing API only reports `pending`, `running`, `completed`, and `failed`. Add durable stage information suitable for polling, for example:

- preparing
- searching
- ranking
- reading selected papers
- building landscape
- verifying gaps
- finalizing

Use actual pipeline boundaries rather than fake timer-based progress. Expose useful counts where available, but do not invent precise percentages. A browser refresh must not lose job status. Polling is sufficient for this milestone; do not add WebSockets unless genuinely necessary.

Maintain the existing bounded worker behavior. Add per-principal concurrency limits and friendly `429`/quota responses. Active provider calls still need not be force-cancelled if that cannot be done safely.

## Pages and routes

Implement at least these pages:

- `/` — public home page
- `/analyze` — separate Quick Search / Full Gap Analysis form
- `/analyses/[analysisId]` — progress and completed result
- `/history` — current user or guest’s authorized history
- `/sign-in`
- `/sign-up`
- `/profile`
- `/pricing`
- `/admin` — administrator only
- appropriate not-found, unauthorized, quota, empty, loading, and failure states

### Home page

The home page must not contain the working search form. It should include:

- A clear hero explaining Research GAP in one sentence
- A short product introduction
- A “How it works” section
- Evidence-backed feature highlights
- A visible limitation statement: the system helps investigate possible gaps but does not prove global novelty or replace a systematic review
- A call to action linking to `/analyze`
- An About Me section using this editable initial copy:

  > I’m Temuujin, a university student focused on artificial intelligence and software engineering. I built Research GAP to make literature exploration more structured, transparent, and useful for researchers developing new ideas.

- Keep personal copy in one obvious content/config file so the owner can edit it without searching through components.
- Footer links for privacy, terms, pricing, contact, and project information. Legal pages may start from clearly labeled owner-reviewed templates; never claim they were reviewed by a lawyer.

### Analyze page

- Large research-idea text area with validation and examples
- Clear selection between Quick Search and Full Gap Analysis
- Advanced options collapsed by default
- Full-text toggle with a cost/latency explanation
- Remaining credits displayed for signed-in users
- Exact guest/free limitation displayed before submission
- Sign-in/upgrade prompt when the selected operation is not allowed
- Accessible submission, pending, rate-limited, validation, and provider-error states

## Results experience

A full analysis should compute and retain the complete result, but the UI must use progressive disclosure rather than dumping everything onto one page.

The initial result view must answer: **How well studied is this idea, and what evidence supports that conclusion?**

Provide these sections or tabs, using the actual backend result fields:

1. **Overview**
   - User-facing assessment: `well_studied`, `uncertain`, or `promising_gap`
   - Plain-language qualification
   - Coverage limitations
   - Papers retrieved/analyzed
   - Abstract/full-text coverage
   - Completion time and analysis mode

2. **Research gaps**
   - Candidate cards
   - Trigger/pattern and landscape basis
   - Supporting papers/evidence
   - Counterexamples
   - Verification result and queries
   - Expandable “Why was this suggested?” explanation

3. **Relevant papers**
   - Search, sort, and useful filters
   - Title, authors, year, DOI/OpenAlex link, abstract preview, relevance components, and full-text availability
   - Retrieval provenance available without overwhelming the default card

4. **Evidence**
   - Research objectives/problems, populations/settings, methods, method families, datasets, sample sizes, baselines, metrics, outcomes, constraints, limitations, and future work
   - Every evidence item traceable to its paper and evidence text/section when available
   - Missing extraction must be labeled “not extracted” rather than “not reported”

5. **Literature landscape**
   - Frequencies, combinations, coverage, and conflicts
   - Add only a few useful responsive charts; retain an accessible textual/table equivalent

6. **Search and coverage**
   - Generated queries and origins
   - Retrieval routes and failures
   - Missing fields and full-text coverage/failure reasons
   - Search limitations

7. **Technical details**
   - Configuration snapshot, models, timings, cache/work metrics, and raw JSON download

Quick Search uses a smaller result layout focused on relevant papers, search strategy, and limitations. Do not imply it performed gap verification.

Provide export/download of the user’s own completed result in JSON and a readable Markdown report. Generate exports server-side or from validated response data; never allow cross-user export.

## Visual design

Create an original, modern research-product interface. Do not copy another company’s site or assets.

Direction:

- Restrained academic/developer-tool aesthetic
- Warm off-white light theme and deep charcoal dark theme
- Indigo/electric violet primary accent
- Teal for supported positive evidence, amber for uncertainty, red for errors/insufficient coverage
- Clean typography, strong hierarchy, soft borders, modest shadows, and restrained corner radii
- Responsive desktop, tablet, and mobile layouts
- Subtle stage/status motion that respects reduced-motion preferences
- Excellent keyboard focus, semantic headings, labels, contrast, and screen-reader text
- Dense research data should remain readable; do not wrap every sentence in a huge card

Avoid generic AI visual clichés: excessive gradients, glowing blobs, glassmorphism everywhere, fake confidence gauges, chat bubbles as the primary workflow, and decorative animations that slow down research work.

Use a compact application shell for authenticated pages with navigation for New Analysis, History, Pricing/Usage, and Profile. Keep administrator navigation separate and role-gated.

## Payments and bank payouts

Implement Stripe in test mode with:

- Server-created Checkout Sessions
- Server-created customer portal sessions where supported
- Verified webhook signatures using the raw request body
- Persistent Stripe customer/subscription/event identifiers
- Unique constraint/idempotency protection for processed event IDs
- A configurable Stripe Price ID
- Reconciliation-safe subscription status updates
- No card-data handling by this application
- Test documentation using Stripe test cards and local webhook forwarding

Do not put a bank account number in source code or the application database. Bank payout details must be entered only through the payment provider’s official secure dashboard/onboarding flow.

The owner lives/studies in Korea and is Mongolian, but do not assume which country, legal entity, tax residence, or bank account will be used for the business. Payment-provider availability and payout requirements depend on those facts and can change. The owner setup report must instruct the owner to verify current eligibility using official provider and government sources before enabling live payments. If Stripe cannot directly onboard the eventual business location, document that limitation and describe provider-abstracted alternatives without silently switching the implementation or giving legal/tax conclusions.

## Required owner report

Create `docs/MILESTONE_9_OWNER_SETUP.md`. This is mandatory and must be written for a non-expert owner. Separate clearly:

- What Codex completed
- What remains in local/test mode
- What the owner must do before local use
- What the owner must do before deployment
- What the owner must do before accepting real money
- What cannot be completed without external accounts or identity verification

Include exact commands, dashboard field names, environment-variable names, expected callback/webhook URLs, and verification steps where they are known. Never include real secrets or the administrator plaintext password.

The report must cover:

1. **Local startup**
   - Python/backend installation and migrations
   - Frontend installation and commands
   - Required processes and URLs
   - Test commands

2. **Environment/secrets checklist**
   - OpenAI/OpenAlex settings
   - Managed-auth public/server values
   - Auth callback/site URLs
   - Avatar storage bucket/policies
   - Stripe publishable/secret/webhook/Price identifiers
   - Application URL, allowed origins, cookie/security settings
   - Admin bootstrap email input
   - Which values are public versus secret
   - Secret rotation and a leak-response checklist

3. **Authentication/email setup**
   - Create/configure the auth project
   - Verification/reset email templates
   - Development and production redirects
   - Custom SMTP/sender-domain setup if needed
   - How to run the admin bootstrap and verify role protection

4. **Stripe test mode**
   - Create the placeholder $1/month recurring product and Price
   - Put its Price ID in configuration
   - Configure Checkout/customer portal
   - Run local webhook forwarding
   - Test successful, failed, duplicate, renewal, and cancellation events
   - Confirm credit grants from webhooks rather than redirect pages

5. **Connecting a bank account and receiving payouts**
   - Explain that the bank account is connected in the official payment-provider dashboard, never in this codebase
   - Identity/business verification prerequisites
   - Payout currency, schedule, fees, reserves, refunds, disputes, chargebacks, and statement descriptor
   - Test-to-live activation checklist
   - Country/business/bank eligibility verification with links to current official sources
   - A first-live-payment and first-payout reconciliation checklist

6. **Changing the plan later**
   - Exact configuration/code location for replacing $1 and five credits
   - Explain why changing an existing Stripe Price is different from creating a new Price
   - Subscription migration/versioning considerations
   - Do not duplicate business rules across frontend and backend

7. **Cost protection**
   - How to inspect per-stage token/request/cost metrics
   - Set OpenAI project budgets/alerts and application-wide limits
   - Choose a real price only after measuring uncached and cached full-text/abstract analyses
   - Suggested cost worksheet formula including provider cost, payment fees, taxes, refunds, hosting, storage, support margin, and abuse
   - Explicit warning that $1/5 is placeholder test pricing and may lose money

8. **Domain, deployment, and operations**
   - Domain purchase and DNS
   - HTTPS
   - Frontend/backend deployment variables
   - Persistent database/storage and backups
   - Migration/rollback procedure
   - Logging, error tracking, uptime monitoring, health checks, retention cleanup, and restore test
   - Reverse proxy/same-origin recommendation

9. **Legal and ownership checklist**
   - Explain plainly that copyright protection for original code/text/design generally arises automatically, while formal registration procedures and benefits depend on jurisdiction
   - Distinguish copyright, trademark registration, domain ownership, company/business registration, and software licensing
   - Repository license decision and third-party license/attribution audit
   - Terms of Service, Privacy Policy, Cookie Policy if applicable, refund/cancellation policy, acceptable-use policy, and contact information
   - Research-data privacy and retention disclosure
   - OpenAI, OpenAlex, Supabase/auth, Stripe, analytics, email, hosting, and storage disclosures/data-processing considerations
   - Consumer subscription renewal/cancellation disclosures
   - Tax/VAT/GST, invoicing, business registration, and accounting questions to confirm with a qualified professional
   - Do not claim legal review or guaranteed compliance
   - Link to current authoritative government/provider documentation, note the date checked, and mark jurisdiction-dependent items for professional review

10. **Pre-launch checklist**
    - Replace placeholder pricing
    - Verify unit economics
    - Remove test-mode banners only after live configuration is intentional
    - Run security, ownership, quota, webhook, backup/restore, accessibility, responsive, and end-to-end checks
    - Verify no secrets or generated administrator credentials are committed

## Configuration and documentation

- Add safe placeholders to `.env.example`; never write secrets there.
- Keep backend business rules authoritative and expose only display-safe plan information to the frontend.
- Update `README.md` with concise Milestone 9 startup instructions and link to the owner setup report.
- Update the Milestone status in `IMPORTANT.md` only after acceptance checks pass.
- Document all new migrations and any compatibility implications.
- Add a small editable site-content/config module for About Me, contact, and external links.

## Security and privacy requirements

- Validate all public input with bounded sizes.
- Escape/safely render paper and evidence text; never inject retrieved HTML.
- Add appropriate security headers.
- Do not leak internal exception details to clients.
- Apply request, login, guest, export, Checkout, and analysis rate limits where appropriate.
- Verify Stripe webhook signatures and auth tokens server-side.
- Use database transactions/constraints for ownership, ledger, idempotency, and quota invariants.
- Redact secrets and tokens from logs and error tracking.
- Avoid logging full unpublished research ideas by default; document the chosen logging policy.
- Never send provider keys to the browser.
- Do not store user-supplied OpenAI keys in this milestone.
- Provide account/data deletion behavior and guest retention cleanup.
- Run dependency/security checks available to the project, but do not blindly rewrite locked versions without need.

## Tests and acceptance criteria

Add deterministic tests covering at least:

### Backend

- Guest Quick Search never initializes or calls paid OpenAI components
- Guest limit and expiry behavior
- Verified free user receives exactly two lifetime full-analysis credits
- Third full analysis is rejected before provider work starts
- Concurrent requests cannot overspend one remaining credit
- Credit reservation settles on success and refunds/releases on failure
- Users cannot list, read, delete, claim, or export another user’s analysis
- Guest session isolation and safe guest-to-account claim
- Legacy records are not exposed
- Unverified user cannot spend full-analysis credit
- Admin authorization is server-enforced and cannot be self-assigned
- Admin credit adjustment creates an audit entry
- Stripe webhook signature rejection
- Duplicate and out-of-order webhook idempotency
- Checkout success redirect alone grants no credits
- Successful paid billing cycle grants exactly the configured allowance once
- Cancellation/payment failure behavior
- Avatar type/size validation
- Existing Milestone 1–8 API/CLI behavior remains valid

### Frontend

- Home page and About Me content render without a search form
- `/analyze` contains the real analysis form
- Authenticated, guest, free-limit, paid, and admin navigation states
- Accessible validation and quota errors
- Polling stops on terminal status and survives a refresh
- Complete result tabs render from representative real-shaped fixtures
- Missing evidence is labeled correctly
- Quick Search does not claim verification
- Mobile and desktop critical flows
- Test-mode pricing is visibly labeled

### End-to-end smoke tests

- Guest Quick Search
- Sign-up/sign-in using a test/fake auth boundary
- Two free full analyses followed by a blocked third request using fake pipeline providers
- Stripe test webhook grants five credits exactly once
- Paid full analysis and result retrieval using fakes
- Admin login/authorization bootstrap path

All automated tests must use fakes and local fixtures. They must not spend money, send real emails, create real payments, or depend on live provider availability.

Run and report:

- Existing Python test suite
- New backend tests
- Frontend lint
- Type checking
- Frontend unit/component tests
- Production frontend build
- End-to-end tests or clearly documented deterministic smoke tests

Inspect the rendered UI at desktop and mobile sizes and fix obvious overflow, unreadable dense data, broken loading states, focus problems, and contrast issues before declaring completion.

## Working rules

- Implement the milestone; do not stop after scaffolding.
- Keep functions and signatures compact when readability allows; do not spread trivial parameters over excessive lines.
- Prefer simple, typed boundaries over unnecessary classes.
- Do not silently change scientific meanings, labels, ranking formulas, extraction behavior, or verification rules.
- Do not delete existing functionality merely to simplify the web implementation.
- Preserve dirty-worktree changes and avoid destructive Git commands.
- Do not commit, push, deploy, enable live payments, purchase a domain, create legal filings, or contact external people unless explicitly authorized.
- If owner credentials are required, complete everything possible, report the precise blocker, and provide the next command. Do not weaken security to make a demo appear complete.

## Final response requirements

When finished, return a concise but complete handoff containing:

1. What was implemented.
2. Important architecture and security decisions.
3. Every test/build command run and its result.
4. Anything incomplete or blocked, stated plainly.
5. A link/path to `docs/MILESTONE_9_OWNER_SETUP.md` and a reminder that it contains all owner actions.
6. The exact local startup commands.
7. Whether the administrator account was actually created and verified.
8. If created, the generated administrator username and one-time password, with a warning to save it in a password manager and rotate it. Never claim they work unless creation was verified, and never commit them.
9. If not created, the exact bootstrap command to create it once the documented authentication variables are configured.
10. The exact location/configuration for changing the placeholder price and paid credit allowance later.

Do not say Milestone 9 is complete merely because pages render. It is complete only when ownership isolation, quota enforcement, admin authorization, payment webhook idempotency, result views, tests, documentation, and the owner setup report are all implemented and verified as far as the available external configuration permits.
