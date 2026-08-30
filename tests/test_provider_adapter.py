"""Tests for the provider adapter — focused on the default `local` provider.

All run offline: the local HTTP call is monkeypatched, so no server (and no
network) is needed.
"""
import json
import urllib.error

import pytest

from src import provider_adapter
from src.provider_adapter import ProviderError, complete, resolve_provider


# --- provider resolution --------------------------------------------------- #

def test_default_provider_is_local(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    assert resolve_provider() == "local"


def test_ollama_alias_maps_to_local():
    assert resolve_provider("ollama") == "local"


def test_explicit_provider_overrides_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "local")
    assert resolve_provider("mock") == "mock"


# --- the local HTTP call (monkeypatched) ----------------------------------- #

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_local_parses_openai_style_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(provider_adapter.urllib.request, "urlopen", fake_urlopen)
    out = complete("system prompt", "user prompt", provider="local", model="phi4-mini")

    assert out == '{"ok": true}'
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "phi4-mini"
    # both messages are forwarded
    assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user"]


def test_local_uses_env_base_url_and_model(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(provider_adapter.urllib.request, "urlopen", fake_urlopen)

    complete("s", "u", provider="local")  # no explicit model -> from env
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["body"]["model"] == "qwen2.5:7b"


def test_local_raises_clear_error_when_unreachable(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(provider_adapter.urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError) as excinfo:
        complete("s", "u", provider="local")
    msg = str(excinfo.value)
    assert "Could not reach" in msg and "ollama" in msg.lower()


def test_local_raises_on_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(
        provider_adapter.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp({"unexpected": "shape"}),
    )
    with pytest.raises(ProviderError):
        complete("s", "u", provider="local")
