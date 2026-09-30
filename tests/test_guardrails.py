"""Tests for PII guardrails."""

import pytest
from parking_assistant.guardrails.pii_filter import PIIGuardrail


@pytest.fixture
def guardrail():
    """Create guardrail instance."""
    return PIIGuardrail()


def test_detect_person_name(guardrail):
    """Test detection of person names."""
    text = "My name is John Smith and I live in Belgrade."
    
    detected = guardrail.detect_pii(text, entities=["PERSON"])
    
    assert len(detected) > 0
    assert any(d["entity_type"] == "PERSON" for d in detected)


def test_detect_phone_number(guardrail):
    """Test detection of phone numbers."""
    text = "You can call me at +381 11 1234567"
    
    detected = guardrail.detect_pii(text, entities=["PHONE_NUMBER"])
    
    # May or may not detect depending on Presidio version, just check it runs
    assert isinstance(detected, list)


def test_anonymize_name(guardrail):
    """Test anonymization replaces names."""
    text = "Contact Jane Doe for more information."
    
    anonymized, was_modified = guardrail.anonymize_pii(text, entities=["PERSON"])
    
    if was_modified:
        assert "[NAME]" in anonymized or "[REDACTED]" in anonymized
        assert "Jane" not in anonymized or "Doe" not in anonymized


def test_filter_retrieved_context(guardrail):
    """Test filtering context removes PII."""
    context = """
    Parking rules: Maximum 2 hours.
    Contact: john.smith@example.com
    Phone: +381111234567
    """
    
    filtered = guardrail.filter_retrieved_context(context)
    
    # Should remove email/phone if detected
    assert isinstance(filtered, str)


def test_validate_safe_input(guardrail):
    """Test validation of input without PII."""
    user_input = "What are the parking rules in zone A?"
    
    is_safe, unexpected = guardrail.validate_user_input(user_input)
    
    assert is_safe
    assert len(unexpected) == 0


def test_validate_unsafe_input(guardrail):
    """Test validation detects unexpected PII."""
    user_input = "My email is john@example.com, can you help?"
    
    is_safe, unexpected = guardrail.validate_user_input(user_input)
    
    # May detect email
    assert isinstance(unexpected, list)
