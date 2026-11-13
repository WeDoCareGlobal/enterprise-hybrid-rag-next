"""PII/PHI redaction layer for Enterprise Hybrid RAG.

Redacts personally identifiable information (PII) and protected health
information (PHI) on two layers:

* Ingestion layer -- documents are redacted before they enter the BM25/Qdrant
  indexes, so secrets never land in the retrieval corpus.
* Retrieval layer -- retrieved documents are redacted before they leave the
  service, so the caller / LLM never sees raw PII/PHI.

Modes:
    REDACT -- replace match with [REDACTED:<TYPE>]
    HASH   -- replace match with [HASHED:<TYPE>:<sha256 prefix>]
    NONE   -- pass through unchanged (findings are still reported)

All patterns are compiled once at import time. Credit-card matches are
Luhn-validated to limit false positives on version numbers, dates, etc.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class RedactionFinding:
    """Single redaction match."""

    type: str
    start: int
    end: int
    value: str  # raw matched text (only kept when store_raw_values=True)


@dataclass
class RedactedText:
    """Outcome of redacting a single string."""

    result: str
    findings: list[RedactionFinding] = field(default_factory=list)
    redacted_count: int = 0


# ---------------------------------------------------------------------------
# Patterns (compiled once, case-insensitive where sensible)
# ---------------------------------------------------------------------------

_EMAIL = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?!\d)"
)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}[\dA-Z]{7,}[A-Z0-9]{1,13}\b")
_MRN = re.compile(
    r"(?i)\b(?:mrn|medical.?record|patient.?id|health.?record)[:\s#]*\d{6,}\b"
)
_API_KEY = re.compile(
    r"\b(?:sk|pk)_(?:live|test|sandbox)_[0-9A-Za-z]{16,}\b"
    r"|\bsk-[0-9A-Za-z]{20,}\b"
    r"|\bghp_[0-9A-Za-z]{36,}\b"
)
_DOB = re.compile(r"(?<!\d)\d{2}[./-]\d{2}[./-]\d{4}(?!\d)")
_IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b")
_CARD_RAW = re.compile(r"(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)")

# Order matters: more specific / higher-sensitivity patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API_KEY", _API_KEY),
    ("IBAN", _IBAN),
    ("SSN", _SSN),
    ("PHONE", _PHONE),
    ("EMAIL", _EMAIL),
    ("MRN", _MRN),
    ("CREDIT_CARD", _CARD_RAW),  # Luhn-validated below
    ("DATE_OF_BIRTH", _DOB),
    ("IP_ADDRESS", _IPV4),
]


def _luhn_valid(card: str) -> bool:
    """Return True if *card* passes the Luhn checksum."""
    digits = [int(ch) for ch in card if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _hash_token(value: str, prefix: int = 10) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:prefix]


class RedactionEngine:
    """PII/PHI redaction engine.

    Parameters
    ----------
    enabled:
        When False all methods pass text through unchanged.
    mode:
        ``REDACT`` / ``HASH`` / ``NONE``.
    store_raw_values:
        Keep raw matched values in findings (audit use only; disabled by
        default so secrets never linger in memory/logs longer than needed).
    custom_patterns:
        Extra ``(name, compiled_regex)`` pairs applied after built-ins.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "REDACT",
        store_raw_values: bool = False,
        custom_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ) -> None:
        if mode not in ("REDACT", "HASH", "NONE"):
            raise ValueError(f"Invalid redaction mode: {mode!r}")
        self.enabled = enabled
        self.mode = mode
        self.store_raw_values = store_raw_values
        self._patterns = list(_PATTERNS) + list(custom_patterns or [])

    # -- public API -------------------------------------------------------

    def redact(self, text: str) -> RedactedText:
        """Redact *text* and return the result plus findings."""
        if not self.enabled or not text:
            return RedactedText(result=text)

        # Collect all matches across patterns, then resolve overlaps
        # (longest match wins; earlier start wins ties).
        matches: list[tuple[int, int, str, str]] = []  # start, end, type, value
        for ptype, pattern in self._patterns:
            for m in pattern.finditer(text):
                start, end = m.span()
                value = m.group(0)
                if ptype == "CREDIT_CARD" and not _luhn_valid(value):
                    continue
                # Skip version-like false positives (e.g. "3.14.9").
                if ptype == "IP_ADDRESS" and re.fullmatch(r"\d+(\.\d+)+", value):
                    parts = value.split(".")
                    if len(parts) == 4 and all(len(p) == 1 for p in parts):
                        continue
                matches.append((start, end, ptype, value))

        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        kept: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, ptype, value in matches:
            if start < last_end:
                continue  # overlap with a longer/earlier match
            kept.append((start, end, ptype, value))
            last_end = end

        if not kept:
            return RedactedText(result=text)

        findings: list[RedactionFinding] = []
        out_parts: list[str] = []
        cursor = 0
        for start, end, ptype, value in kept:
            out_parts.append(text[cursor:start])
            replacement = self._replacement(ptype, value)
            out_parts.append(replacement)
            cursor = end
            findings.append(
                RedactionFinding(
                    type=ptype,
                    start=start,
                    end=end,
                    value=value if self.store_raw_values else "",
                )
            )
        out_parts.append(text[cursor:])

        return RedactedText(
            result="".join(out_parts),
            findings=findings,
            redacted_count=len(findings),
        )

    def redact_document(self, document: dict[str, Any]) -> dict[str, Any]:
        """Redact string fields of a document dict (text + metadata strings)."""
        redacted: dict[str, Any] = {}
        for key, value in document.items():
            if isinstance(value, str):
                redacted[key] = self.redact(value).result
            elif isinstance(value, dict):
                redacted[key] = {
                    k: self.redact(v).result if isinstance(v, str) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact(item).result if isinstance(item, str) else item
                    for item in value
                ]
            else:
                redacted[key] = value
        return redacted

    # -- convenience helpers ----------------------------------------------

    def _replacement(self, ptype: str, value: str) -> str:
        if self.mode == "NONE":
            return value
        if self.mode == "REDACT":
            return f"[REDACTED:{ptype}]"
        return f"[HASHED:{ptype}:{_hash_token(value)}]"


# Singleton used by the API layer. Override via configure_redaction().
_default_engine: RedactionEngine | None = None


def get_redaction_engine() -> RedactionEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RedactionEngine()
    return _default_engine


def configure_redaction(
    *,
    enabled: bool | None = None,
    mode: str | None = None,
    store_raw_values: bool | None = None,
    custom_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
) -> RedactionEngine:
    """Build (or rebuild) the default redaction engine from env-ish params."""
    engine = get_redaction_engine()
    if enabled is not None:
        engine.enabled = enabled
    if mode is not None:
        engine.mode = mode
    if store_raw_values is not None:
        engine.store_raw_values = store_raw_values
    if custom_patterns is not None:
        engine._patterns = list(_PATTERNS) + list(custom_patterns)  # noqa: SLF001
    _default_engine = engine
    return engine
