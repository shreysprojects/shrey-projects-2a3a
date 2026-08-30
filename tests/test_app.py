"""Tests for the local web UI's API handlers (app.py).

The handlers are plain functions, so these run fully offline on the 'mock'
provider — no server, no network, no model. They prove status codes, error
shapes, and that results serialise to JSON-safe dicts.
"""
import json

from app import handle_check, handle_cleanup, handle_info


def test_cleanup_happy_path_mock():
    status, body = handle_cleanup({"text": "app keeps crashing", "provider": "mock"})
    assert status == 200
    assert "short_summary" in body["result"]
    assert body["elapsed_seconds"] >= 0
    json.dumps(body)  # everything must be JSON-serialisable


def test_cleanup_blank_text_is_400():
    status, body = handle_cleanup({"text": "   ", "provider": "mock"})
    assert status == 400
    assert "error" in body


def test_cleanup_unknown_provider_is_502():
    status, body = handle_cleanup({"text": "hello", "provider": "definitely-not-real"})
    assert status == 502
    assert "error" in body


def test_check_happy_path_mock_never_auto_sends():
    status, body = handle_check({
        "english": "Your order has shipped.",
        "spanish": "Su pedido ha sido enviado.",
        "provider": "mock",
    })
    assert status == 200
    # mock can't judge, so the conservative default must not be "Send"
    assert body["result"]["send_recommendation"] == "Review First"
    json.dumps(body)


def test_check_english_in_spanish_field_is_400():
    status, body = handle_check({
        "english": "Please send us your order number.",
        "spanish": "Please send us your order number.",
        "provider": "mock",
    })
    assert status == 400
    assert "Spanish" in body["error"]


def test_check_gate_escalates_even_on_mock():
    # Mock says Review First; the deterministic gate must escalate an
    # unlicensed guarantee to Do Not Send — proving the gate runs in the UI path.
    status, body = handle_check({
        "english": "We will look into your billing question.",
        "spanish": "Le garantizamos un reembolso completo de inmediato.",
        "provider": "mock",
    })
    assert status == 200
    assert body["result"]["send_recommendation"] == "Do Not Send"
    assert any("[safety gate]" in p for p in body["result"]["risky_phrases"])


def test_info_reports_models():
    status, body = handle_info()
    assert status == 200
    assert {"provider", "cleanup_model", "translation_model"} <= set(body)
