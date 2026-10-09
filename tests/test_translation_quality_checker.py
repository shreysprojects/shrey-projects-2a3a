"""Tests for Project 3A (Translation Quality Checker).

These run entirely on the 'mock' provider (no API key, no network), plus a few
that drive the schema and the retry path directly. The mock provider can't judge
translation quality, so the end-to-end tests assert on the things that ARE
deterministic offline: input validation (blank / not-Spanish), schema shape, the
conservative "never auto-Send" default, and enum/clamp normalisation.
"""
import json

import pytest

from src import translation_quality_checker
from src.schemas import (
    AccuracyRating,
    SendRecommendation,
    ToneCheck,
    TranslationQualityResult,
)
from src.translation_quality_checker import (
    EmptyInputError,
    NotSpanishError,
    SchemaValidationError,
    check_translation,
)

GOOD_EN = "We are sorry your order was delayed. Please send us your order number."
GOOD_ES = "Lamentamos que su pedido se haya retrasado. Envíenos su número de pedido."


def _valid_payload(**overrides):
    base = {
        "short_summary": "Safe to send: accurate and professional.",
        "accuracy_rating": "High",
        "confidence_score": 90,
        "tone_check": "Professional",
        "risky_phrases": [],
        "missing_meaning": [],
        "added_meaning": [],
        "suggested_correction": "",
        "back_translation": "We are sorry your order was delayed. Send us your order number.",
        "send_recommendation": "Send",
        "explanation": "Accurate and professional.",
    }
    base.update(overrides)
    return base


# --- input validation (spec rule 15, test case 5) -------------------------- #

def test_blank_spanish_raises():
    with pytest.raises(EmptyInputError):
        check_translation(GOOD_EN, "   ", provider="mock")


def test_blank_english_raises():
    with pytest.raises(EmptyInputError):
        check_translation("", GOOD_ES, provider="mock")


def test_english_in_spanish_field_is_rejected():
    # Operator pasted the English message into the Spanish field.
    with pytest.raises(NotSpanishError):
        check_translation(GOOD_EN, "Please send us your order number now.", provider="mock")


def test_real_spanish_is_accepted():
    result = check_translation(GOOD_EN, GOOD_ES, provider="mock")
    assert isinstance(result, TranslationQualityResult)


# --- UI-facing short summary (task #1) ------------------------------------- #

def test_result_has_short_summary():
    result = check_translation(GOOD_EN, GOOD_ES, provider="mock")
    assert isinstance(result.short_summary, str) and result.short_summary.strip()


def test_summary_only_prints_single_clean_line(capsys):
    # --summary-only is the clean UI output: exactly one line, no JSON/debug.
    exit_code = translation_quality_checker.main(
        [GOOD_EN, GOOD_ES, "--provider", "mock", "--summary-only"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "{" not in out and "}" not in out


# --- end-to-end via the mock provider -------------------------------------- #

def test_mock_never_auto_sends():
    # The offline stub can't judge quality, so it must not recommend "Send".
    result = check_translation(GOOD_EN, GOOD_ES, provider="mock")
    assert result.send_recommendation != SendRecommendation.SEND
    assert result.send_recommendation == SendRecommendation.REVIEW_FIRST


def test_result_has_all_required_fields():
    result = check_translation(GOOD_EN, GOOD_ES, provider="mock")
    data = json.loads(result.model_dump_json())
    for field in [
        "short_summary", "accuracy_rating", "confidence_score", "tone_check",
        "risky_phrases", "missing_meaning", "added_meaning", "suggested_correction",
        "back_translation", "send_recommendation", "explanation",
    ]:
        assert field in data


# --- schema behaviour ------------------------------------------------------ #

def test_enums_accept_any_casing():
    result = TranslationQualityResult.model_validate(
        _valid_payload(accuracy_rating="high", tone_check="too harsh", send_recommendation="do not send")
    )
    assert result.accuracy_rating == AccuracyRating.HIGH
    assert result.tone_check == ToneCheck.TOO_HARSH
    assert result.send_recommendation == SendRecommendation.DO_NOT_SEND


def test_underscored_recommendation_is_normalised():
    result = TranslationQualityResult.model_validate(
        _valid_payload(send_recommendation="do_not_send")
    )
    assert result.send_recommendation == SendRecommendation.DO_NOT_SEND


def test_unknown_enum_values_fall_back_conservatively():
    result = TranslationQualityResult.model_validate(
        _valid_payload(accuracy_rating="banana", send_recommendation="maybe")
    )
    # Unknown accuracy -> Low (assume worst); unknown send -> Review First (never auto-Send).
    assert result.accuracy_rating == AccuracyRating.LOW
    assert result.send_recommendation == SendRecommendation.REVIEW_FIRST


def test_confidence_score_is_clamped():
    assert TranslationQualityResult.model_validate(_valid_payload(confidence_score=150)).confidence_score == 100
    assert TranslationQualityResult.model_validate(_valid_payload(confidence_score=-5)).confidence_score == 0


def test_string_where_list_expected_is_wrapped():
    result = TranslationQualityResult.model_validate(
        _valid_payload(added_meaning="refund promise")
    )
    assert result.added_meaning == ["refund promise"]


def test_null_optional_text_becomes_empty_string():
    result = TranslationQualityResult.model_validate(
        _valid_payload(suggested_correction=None)
    )
    assert result.suggested_correction == ""


# --- retry / validation path ----------------------------------------------- #

def test_non_json_response_raises_after_retries(monkeypatch):
    monkeypatch.setattr(
        translation_quality_checker, "complete", lambda *a, **k: "this is not json at all"
    )
    with pytest.raises(SchemaValidationError):
        check_translation(GOOD_EN, GOOD_ES, provider="openai", max_retries=1)


def test_recovers_when_second_attempt_is_valid(monkeypatch):
    responses = iter(["garbage, no json", "valid"])

    def fake_complete(system, user, **kwargs):
        text = next(responses)
        return json.dumps(_valid_payload()) if text == "valid" else text

    monkeypatch.setattr(translation_quality_checker, "complete", fake_complete)
    result = check_translation(GOOD_EN, GOOD_ES, provider="openai", max_retries=2)
    assert result.send_recommendation == SendRecommendation.SEND
