"""Tests for Project 2A.

These run entirely on the 'mock' provider (no API key, no network), plus a few
that drive the schema and the retry path directly.
"""
import pytest

from src import service_request_cleanup
from src.schemas import IssueType, ServiceRequestResult
from src.service_request_cleanup import (
    EmptyInputError,
    SchemaValidationError,
    clean_service_request,
)


def _valid_payload(**overrides):
    base = {
        "cleaned_description": "The customer cannot log in.",
        "short_summary": "Customer cannot log in.",
        "issue_type": "Login",
        "customer_impact": "Customer is unable to access their account.",
        "technical_details": [],
        "missing_information": [],
        "suggested_next_step": "Verify the customer's email address.",
        "tone_adjustment": "Neutralised.",
        "ml_tags": [],
        "confidence_score": 80,
    }
    base.update(overrides)
    return base


# --- end-to-end via the mock provider ------------------------------------- #

def test_blank_input_raises():
    with pytest.raises(EmptyInputError):
        clean_service_request("   ", provider="mock")


def test_valid_input_returns_schema_object():
    result = clean_service_request(
        "cust cant login password reset not working very upset", provider="mock"
    )
    assert isinstance(result, ServiceRequestResult)
    assert result.issue_type == IssueType.LOGIN
    assert 0 <= result.confidence_score <= 100


def test_very_short_input_gets_low_confidence():
    result = clean_service_request("app keeps crashing", provider="mock")
    assert result.confidence_score <= 40


# --- UI-facing short summary (task #1) ------------------------------------- #

def test_summary_only_prints_single_clean_line(capsys):
    # --summary-only is the clean UI output: exactly one line, no JSON/debug.
    exit_code = service_request_cleanup.main(
        ["cust cant login password reset not working", "--provider", "mock", "--summary-only"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "{" not in out and "}" not in out


def test_issue_type_is_always_in_the_allowed_set():
    result = clean_service_request(
        "totally unrelated rambling about the weather today", provider="mock"
    )
    assert result.issue_type in set(IssueType)


# --- schema behaviour ------------------------------------------------------ #

def test_lowercase_issue_type_is_normalised():
    result = ServiceRequestResult.model_validate(_valid_payload(issue_type="login"))
    assert result.issue_type == IssueType.LOGIN


def test_unknown_issue_type_becomes_other():
    result = ServiceRequestResult.model_validate(_valid_payload(issue_type="banana"))
    assert result.issue_type == IssueType.OTHER


def test_confidence_score_is_clamped():
    assert ServiceRequestResult.model_validate(_valid_payload(confidence_score=150)).confidence_score == 100
    assert ServiceRequestResult.model_validate(_valid_payload(confidence_score=-5)).confidence_score == 0


def test_string_where_list_expected_is_wrapped():
    result = ServiceRequestResult.model_validate(
        _valid_payload(technical_details="single detail")
    )
    assert result.technical_details == ["single detail"]


# --- retry / validation path ---------------------------------------------- #

def test_non_json_response_raises_after_retries(monkeypatch):
    monkeypatch.setattr(
        service_request_cleanup, "complete", lambda *a, **k: "this is not json at all"
    )
    with pytest.raises(SchemaValidationError):
        clean_service_request("some note", provider="openai", max_retries=1)


def test_recovers_when_second_attempt_is_valid(monkeypatch):
    responses = iter(["garbage, no json", '{"x": 1}'])  # 1st bad shape, 2nd valid-ish

    def fake_complete(system, user, **kwargs):
        text = next(responses)
        if text == '{"x": 1}':
            # return a full valid object on the retry
            import json

            return json.dumps(_valid_payload())
        return text

    monkeypatch.setattr(service_request_cleanup, "complete", fake_complete)
    result = clean_service_request("some note", provider="openai", max_retries=2)
    assert result.issue_type == IssueType.LOGIN
