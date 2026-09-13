"""Fail CI on likely production credentials outside explicit test fixtures."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    re.compile(r"\bsk_live_[A-Za-z0-9]{16,}"),
    re.compile(r"\brk_live_[A-Za-z0-9]{16,}"),
    re.compile(r"\bwhsec_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def main() -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    findings: list[str] = []
    for relative in tracked:
        if not relative or relative.startswith(("tests/", "web/node_modules/", "web/.next/")):
            continue
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern.search(text) for pattern in PATTERNS):
            findings.append(relative)
    if findings:
        print("Possible committed production secret(s): " + ", ".join(sorted(findings)))
        return 1
    print("No likely committed production credentials found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
