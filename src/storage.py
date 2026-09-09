"""Validated profile-image storage boundary."""

from __future__ import annotations

from typing import Protocol
from urllib import error, request


class AvatarError(ValueError):
    pass


class AvatarStorage(Protocol):
    def upload(self, user_id: str, content: bytes, media_type: str) -> str: ...
    def delete(self, user_id: str) -> None: ...


def validate_avatar(content: bytes, *, max_bytes: int = 2_000_000) -> str:
    if not content or len(content) > max_bytes:
        raise AvatarError("Avatar must be between 1 byte and 2 MB.")
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    raise AvatarError("Avatar must be a valid PNG, JPEG, or WebP image.")


class SupabaseAvatarStorage:
    def __init__(self, url: str, service_key: str, bucket: str = "avatars") -> None:
        self.url, self.service_key, self.bucket = url.rstrip("/"), service_key, bucket

    def upload(self, user_id: str, content: bytes, media_type: str) -> str:
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[media_type]
        path = f"{user_id}/profile.{extension}"
        req = request.Request(
            f"{self.url}/storage/v1/object/{self.bucket}/{path}", data=content, method="POST",
            headers={"Authorization": f"Bearer {self.service_key}", "apikey": self.service_key,
                     "Content-Type": media_type, "x-upsert": "true"},
        )
        try:
            with request.urlopen(req, timeout=20):
                pass
        except error.URLError as exc:
            raise AvatarError("Avatar storage is unavailable.") from exc
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{path}"

    def delete(self, user_id: str) -> None:
        # Bucket lifecycle/policies may retain older extensions. Account deletion is
        # documented to require the provider-side folder cleanup policy as a backstop.
        for extension in ("png", "jpg", "webp"):
            path = f"{user_id}/profile.{extension}"
            req = request.Request(
                f"{self.url}/storage/v1/object/{self.bucket}/{path}", method="DELETE",
                headers={"Authorization": f"Bearer {self.service_key}", "apikey": self.service_key},
            )
            try:
                request.urlopen(req, timeout=10).close()
            except error.URLError:
                continue
