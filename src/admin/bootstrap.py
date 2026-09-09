"""One-time managed-auth administrator bootstrap."""

from __future__ import annotations

import argparse
import secrets
import string

from src.auth import AuthenticationError, SupabaseAuthProvider
from src.config import Settings
from src.persistence.database import Database
from src.persistence.security import SecurityRepository


def _password(length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits + "-_.!@#$%"
    while True:
        value = "".join(secrets.choice(alphabet) for _ in range(length))
        if all(any(c in group for c in value) for group in (string.ascii_lowercase, string.ascii_uppercase, string.digits, "!@#$%")):
            return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the one owner-controlled Research GAP administrator")
    parser.add_argument("--email", required=True, help="Owner-controlled login email")
    args = parser.parse_args()
    settings = Settings.from_env()
    web = settings.web
    database = Database(settings.analysis_database_path)
    database.migrate()
    security = SecurityRepository(database, free_credits=web.free_lifetime_credits)
    existing = security.admin_account()
    if existing:
        print(f"Administrator already bootstrapped: {existing['display_name']} ({existing['email']})")
        return 0
    if security.account_by_email(args.email):
        parser.error("That email already belongs to a non-administrator account; use a distinct owner login email")
    if not web.auth_url or not web.auth_service_role_key:
        parser.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured first")
    username = "owner-" + secrets.token_hex(5)
    password = _password()
    provider = SupabaseAuthProvider(
        web.auth_url, audience=web.auth_jwt_audience, issuer=web.auth_jwt_issuer,
        service_role_key=web.auth_service_role_key,
    )
    try:
        identity = provider.create_admin_user(args.email, password, username)
    except AuthenticationError as exc:
        parser.error(str(exc))
    security.sync_account(identity.user_id, email=identity.email, verified=True)
    security.profile_update(identity.user_id, display_name=username)
    security.set_role(identity.user_id, "admin")
    print(f"Administrator created and verified. Username: {username}")
    print(f"One-time password: {password}")
    print("Store it in a password manager now and change it after first login. It was not written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
