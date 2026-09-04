"""Deterministic request and response safety checks that do not require an LLM."""

import re


class GuardrailViolation(ValueError):
    """Raised when a request conflicts with deterministic safety controls."""


_INJECTION_PATTERNS = (
    r"\bignore (all |any |the )?(previous|prior) instructions?\b",
    r"\b(reveal|show|print|repeat) (the )?(system|developer) prompt\b",
    r"\bdisregard (all |any |the )?(previous|prior) instructions?\b",
)


def validate_input(query: str) -> str:
    """Reject unsafe prompt-control requests before they reach routing or models."""
    normalized = " ".join(query.split())
    if len(normalized) > 4_000:
        raise GuardrailViolation("Query exceeds the 4,000 character limit.")
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _INJECTION_PATTERNS):
        raise GuardrailViolation("This request attempts to override protected agent instructions.")
    return normalized


def remove_reasoning_leak(text: str) -> str:
    """Remove common accidental reasoning prefixes from model output."""
    patterns = (
        r"^here(?:'|’)s a thinking process:.*?(?=^## |\A(?![\s\S]))",
        r"^analysis:.*?(?=^## |\A(?![\s\S]))",
        r"^let me think.*?(?=^## |\A(?![\s\S]))",
        r"^analyze user input:.*?(?=^## |\A(?![\s\S]))",
    )

    cleaned = text.strip()

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        ).strip()

    return cleaned

def sanitize_output(text: str) -> str:
    cleaned = remove_reasoning_leak(text)

    # Keep your existing sanitization logic below.
    return cleaned
