"""Deterministic request and response safety checks that do not require an LLM."""

import re


class GuardrailViolation(ValueError):
    """Raised when a request conflicts with deterministic safety controls."""


_INJECTION_PATTERNS = (
    r"\bignore (all |any |the )?(previous|prior) instructions?\b",
    r"\b(reveal|show|print|repeat) (the )?(system|developer) prompt\b",
    r"\bdisregard (all |any |the )?(previous|prior) instructions?\b",
)
# Match recognizable provider-key prefixes without storing or logging the values.
_SECRET_PATTERN = re.compile(r"(?:AIza[\w-]{20,}|gsk_[\w-]{20,}|lsv2_[\w-]{20,})")


def validate_input(query: str) -> str:
    """Reject unsafe prompt-control requests before they reach routing or models."""
    normalized = " ".join(query.split())
    if len(normalized) > 4_000:
        raise GuardrailViolation("Query exceeds the 4,000 character limit.")
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _INJECTION_PATTERNS):
        raise GuardrailViolation("This request attempts to override protected agent instructions.")
    return normalized


def sanitize_output(answer: str) -> str:
    """Prevent API-like secrets from being returned to the chat client or traces."""
    return _SECRET_PATTERN.sub("[redacted secret]", answer)
