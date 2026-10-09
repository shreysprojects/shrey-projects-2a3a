"""Tests for the optional semantic (NLI) check layer.

All offline: the real model is never loaded. We test the wiring — the on/off
switch, graceful no-op when the ML stack is absent, and the escalate-only
merge — by monkeypatching the scoring internals. The layer's real accuracy is
measured by the eval harness (samples/evaluation/), not unit tests.
"""
import pytest

from src import nli_check
from src.schemas import SendRecommendation, TranslationQualityResult

EN = "Your subscription is active."
ES = "Su suscripción ha sido suspendida."


def _result(recommendation="Send"):
    return TranslationQualityResult.model_validate({
        "short_summary": "Safe to send: accurate and professional.",
        "accuracy_rating": "High",
        "confidence_score": 90,
        "tone_check": "Professional",
        "risky_phrases": [],
        "missing_meaning": [],
        "added_meaning": [],
        "suggested_correction": "",
        "back_translation": "(bt)",
        "send_recommendation": recommendation,
        "explanation": "Looks fine.",
    })


def _fake_scores(fwd, bwd):
    """Patchable _scores: first call returns fwd, second returns bwd."""
    calls = []

    def scores(premise, hypothesis):
        calls.append(1)
        return fwd if len(calls) == 1 else bwd

    return scores


def test_off_switch_disables_layer(monkeypatch):
    monkeypatch.setenv("NLI_CHECK", "off")
    assert nli_check.is_enabled() is False
    assert nli_check.run_nli_check(EN, ES) == []


def test_missing_stack_is_silent_noop(monkeypatch):
    # Simulate torch/transformers not installed: pipe unavailable.
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: None)
    result = _result("Send")
    assert nli_check.apply_nli_check(result, EN, ES) == result


def test_contradiction_escalates_to_do_not_send(monkeypatch):
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())
    monkeypatch.setattr(nli_check, "_scores", _fake_scores(
        {"entailment": 0.01, "neutral": 0.04, "contradiction": 0.95},
        {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90},
    ))
    gated = nli_check.apply_nli_check(_result("Send"), EN, ES)
    assert gated.send_recommendation is SendRecommendation.DO_NOT_SEND
    assert any("[semantic check]" in p for p in gated.risky_phrases)
    assert gated.short_summary.startswith("Do not send")


def test_one_directional_contradiction_does_not_block(monkeypatch):
    # A contradiction seen in only ONE direction is the false-positive
    # signature measured on clean pairs — it must not hard-block (worst
    # case it lands in the weak-entailment Review First branch).
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())
    monkeypatch.setattr(nli_check, "_scores", _fake_scores(
        {"entailment": 0.05, "neutral": 0.02, "contradiction": 0.93},
        {"entailment": 0.60, "neutral": 0.25, "contradiction": 0.15},
    ))
    gated = nli_check.apply_nli_check(_result("Send"), EN, ES)
    assert gated.send_recommendation is not SendRecommendation.DO_NOT_SEND


def test_weak_entailment_escalates_to_review_first(monkeypatch):
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())
    monkeypatch.setattr(nli_check, "_scores", _fake_scores(
        {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
        {"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05},
    ))
    gated = nli_check.apply_nli_check(_result("Send"), EN, ES)
    assert gated.send_recommendation is SendRecommendation.REVIEW_FIRST


def test_clean_pair_is_untouched(monkeypatch):
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())
    monkeypatch.setattr(nli_check, "_scores", _fake_scores(
        {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01},
        {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01},
    ))
    result = _result("Send")
    assert nli_check.apply_nli_check(result, EN, ES) == result


def test_never_downgrades(monkeypatch):
    # Model already said Do Not Send; a weak-entailment (Review First) finding
    # must not soften it.
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())
    monkeypatch.setattr(nli_check, "_scores", _fake_scores(
        {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
        {"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05},
    ))
    gated = nli_check.apply_nli_check(_result("Do Not Send"), EN, ES)
    assert gated.send_recommendation is SendRecommendation.DO_NOT_SEND


def test_scoring_failure_never_breaks_pipeline(monkeypatch):
    monkeypatch.setattr(nli_check, "_get_pipe", lambda: object())

    def boom(premise, hypothesis):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(nli_check, "_scores", boom)
    result = _result("Send")
    assert nli_check.apply_nli_check(result, EN, ES) == result
