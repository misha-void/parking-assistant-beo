"""Tests for the Presidio PII guardrail."""

import pytest

from parking_assistant.guardrails.pii_filter import PIIGuardrail


@pytest.fixture(scope="module")
def guardrail():
    """One guardrail for the module; loading the spaCy model is slow."""
    return PIIGuardrail()


def test_detects_a_person_name(guardrail):
    detected = guardrail.detect_pii("My name is John Smith.", entities=["PERSON"])
    assert any(item["entity_type"] == "PERSON" for item in detected)


def test_anonymize_replaces_the_name(guardrail):
    anonymized, was_modified = guardrail.anonymize_pii(
        "Contact Jane Doe today.", entities=["PERSON"]
    )
    assert was_modified
    assert "Jane" not in anonymized


def test_context_filter_redacts_an_email(guardrail):
    filtered = guardrail.filter_retrieved_context("Write to john.smith@example.com for help.")
    assert "john.smith@example.com" not in filtered


def test_context_filter_keeps_parking_data(guardrail):
    # Regression: spot counts were once redacted as phone numbers, mangling availability.
    context = "UScE Garage, Bulevar Mihajla Pupina 4 - 1800/2500 spots - 100 RSD/hour"
    assert guardrail.filter_retrieved_context(context) == context


def test_llm_output_filter_redacts_secrets_but_keeps_booking_details(guardrail):
    # The reservation flow legitimately echoes the user's own name and plate back.
    output = "Thanks Misha, plate BG123AB is booked. Invoices go to billing@example.com."
    filtered = guardrail.filter_llm_output(output)
    assert "billing@example.com" not in filtered
    assert "Misha" in filtered
    assert "BG123AB" in filtered
