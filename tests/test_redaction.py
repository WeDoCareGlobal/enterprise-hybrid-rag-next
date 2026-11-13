"""Unit tests for the PII/PHI redaction layer."""

from app.redaction import RedactionEngine, RedactedText, configure_redaction, get_redaction_engine


def _engine() -> RedactionEngine:
    return RedactionEngine(mode="REDACT", store_raw_values=False)


def test_redacts_email():
    eng = _engine()
    out = eng.redact("Contact me at ivanic.peric@example.com or support@firma.ba.")
    assert "ivanic.peric@example.com" not in out.result
    assert "support@firma.ba" not in out.result
    assert out.redacted_count == 2
    assert out.result.count("[REDACTED:EMAIL]") == 2


def test_redacts_phone_and_ssn():
    eng = _engine()
    out = eng.redact("Phone +387 60 123 4567, SSN 123-45-6789.")
    assert "+387 60 123 4567" not in out.result
    assert "123-45-6789" not in out.result


def test_redacts_credit_card_luhn():
    eng = _engine()
    # 4111-1111-1111-1111 passes Luhn.
    out = eng.redact("Card 4111-1111-1111-1111 used for payment.")
    assert "4111-1111-1111-1111" not in out.result
    assert "[REDACTED:CREDIT_CARD]" in out.result


def test_ignores_non_luhn_card_like_numbers():
    eng = _engine()
    out = eng.redact("Version 3.14.0 release notes.")
    assert out.redacted_count == 0
    assert out.result == "Version 3.14.0 release notes."


def test_redacts_iban_and_api_key():
    eng = _engine()
    out = eng.redact("IBAN BA391234567890123456 and sk-live_abcdef0123456789.")
    assert "BA391234567890123456" not in out.result
    assert "sk-live_abcdef0123456789" not in out.result


def test_redacts_mrn_and_dob():
    eng = _engine()
    out = eng.redact("MRN# 987654321, DOB 12.05.1985.")
    assert "987654321" not in out.result
    assert "12.05.1985" not in out.result


def test_hash_mode():
    eng = RedactionEngine(mode="HASH")
    out = eng.redact("Mail ivanic.peric@example.com.")
    assert "ivanic.peric@example.com" not in out.result
    assert "[HASHED:EMAIL:" in out.result


def test_none_mode_passes_through():
    eng = RedactionEngine(mode="NONE")
    out = eng.redact("Mail ivanic.peric@example.com.")
    assert out.result == "Mail ivanic.peric@example.com."
    assert out.redacted_count == 1  # finding still reported


def test_redact_document_metadata():
    eng = _engine()
    doc = {"id": "1", "text": "Patient email ivanic.peric@example.com.", "metadata": {"doctor": "Dr. Ivic, 123-45-6789"}}
    redacted = eng.redact_document(doc)
    assert "ivanic.peric@example.com" not in redacted["text"]
    assert "123-45-6789" not in redacted["metadata"]["doctor"]


def test_disabled_engine_passes_through():
    eng = RedactionEngine(enabled=False)
    out = eng.redact("ssn 123-45-6789 and mail a@b.c")
    assert out.result == "ssn 123-45-6789 and mail a@b.c"
    assert out.redacted_count == 0


def test_findings_do_not_store_raw_by_default():
    eng = _engine()
    out = eng.redact("a@b.c")
    assert all(f.value == "" for f in out.findings)


def test_findings_store_raw_when_enabled():
    eng = RedactionEngine(store_raw_values=True)
    out = eng.redact("a@b.co")
    assert out.findings[0].value == "a@b.co"
    assert out.findings[0].type == "EMAIL"


def test_configure_redaction_singleton():
    configure_redaction(mode="HASH")
    eng = get_redaction_engine()
    assert eng.mode == "HASH"
    out = eng.redact("x@y.co")
    assert "[HASHED:EMAIL:" in out.result
    # Reset for other tests.
    configure_redaction(mode="REDACT")
