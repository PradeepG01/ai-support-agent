"""Inbound webhook security: HMAC-SHA256 signatures with timestamp replay protection.

Senders compute:  signature = hex(HMAC_SHA256(secret, f"{timestamp}.{raw_body}"))
and send headers  X-Signature: sha256=<signature>   X-Timestamp: <unix seconds>
"""

from __future__ import annotations

import hashlib
import hmac
import time

MAX_SKEW_SECONDS = 300


def sign(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify(secret: str, timestamp: str | None, signature: str | None, body: bytes, now: float | None = None) -> bool:
    if not timestamp or not signature:
        return False
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    if abs((now or time.time()) - sent_at) > MAX_SKEW_SECONDS:
        return False
    return hmac.compare_digest(sign(secret, timestamp, body), signature)
