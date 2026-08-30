"""Provider adapter layer.

The business logic (cleanup / translation) talks to THIS module, never directly
to a model vendor. To add or swap a provider you change only this file — the
utilities don't change. That's the whole point: no hard dependency on one AI
vendor.

API keys come from environment variables (loaded from a local ``.env`` file by
the CLI). Keys are never hard-coded.

Supported providers
--------------------
  local      DEFAULT. A self-hosted, OpenAI-compatible server (e.g. Ollama).
             No API key, no cloud, no extra pip install (uses the stdlib).
             Configure with LOCAL_BASE_URL / LOCAL_MODEL.          alias: ollama
  mock       offline stub, no key needed — for tests and quick demos only.
  openai     needs OPENAI_API_KEY      (pip install openai)
  anthropic  needs ANTHROPIC_API_KEY   (pip install anthropic)   alias: claude
  google     needs GOOGLE_API_KEY      (pip install google-generativeai) alias: gemini
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    """Raised for a missing key, missing SDK, or unknown provider name."""


DEFAULT_PROVIDER = "local"
_ALIASES = {"claude": "anthropic", "gemini": "google", "gpt": "openai", "ollama": "local"}


def resolve_provider(explicit: str | None = None) -> str:
    """Pick the provider: explicit arg > MODEL_PROVIDER env var > 'mock'."""
    name = (explicit or os.getenv("MODEL_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    return _ALIASES.get(name, name)


def is_offline(name: str) -> bool:
    return resolve_provider(name) == "mock"


def complete(
    system: str,
    user: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    mock_response: dict | None = None,
    response_schema: dict | None = None,
) -> str:
    """Send one prompt to the chosen provider and return the raw text response.

    ``mock_response`` is the offline stand-in returned when provider == 'mock';
    each utility supplies its own (see the cleanup module's offline stub).

    ``response_schema`` is an optional JSON Schema. When given, the local provider
    asks the server to CONSTRAIN output to that schema (Ollama structured outputs
    / llama.cpp grammar), which guarantees valid JSON and — crucially — that enum
    fields only take allowed values. It does not make the model choose the RIGHT
    value, only a valid one; correct choices still depend on the model.
    """
    name = resolve_provider(provider)
    if name == "mock":
        return json.dumps(mock_response if mock_response is not None else {})
    if name == "local":
        return _local(system, user, model, temperature, response_schema)
    if name == "openai":
        return _openai(system, user, model, temperature)
    if name == "anthropic":
        return _anthropic(system, user, model, temperature)
    if name == "google":
        return _google(system, user, model, temperature)
    raise ProviderError(
        f"Unknown provider '{name}'. Use one of: local, mock, openai, anthropic, google."
    )


# --------------------------------------------------------------------------- #
# Provider implementations. SDKs are imported lazily so the module loads (and
# tests run) even when a given SDK isn't installed.
# --------------------------------------------------------------------------- #

def _local(
    system: str,
    user: str,
    model: str | None,
    temperature: float,
    response_schema: dict | None = None,
) -> str:
    """Call a local, OpenAI-compatible model server (default: Ollama).

    No cloud and no API key. Uses only the standard library, so there is no
    extra pip install. Point LOCAL_BASE_URL at any OpenAI-compatible endpoint:
    Ollama (:11434/v1), vLLM (:8000/v1), LM Studio (:1234/v1), etc.
    """
    # 127.0.0.1 (not "localhost") so we don't hit IPv6 ::1 while Ollama binds IPv4.
    base_url = (os.getenv("LOCAL_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
    model_name = model or os.getenv("LOCAL_MODEL") or "phi4-mini"

    # Qwen3 hybrid models emit a chain-of-thought by default, which wastes the
    # tight CPU latency budget (and can blow the timeout). '/no_think' disables it.
    # Kept here (not in the task prompts) so vendor quirks stay isolated to this layer.
    if "qwen3" in model_name.lower():
        system = f"{system}\n/no_think"

    # With a schema: constrain output (guarantees valid JSON + in-enum values).
    # Without: fall back to plain JSON mode (older behaviour).
    if response_schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": response_schema, "strict": True},
        }
    else:
        response_format = {"type": "json_object"}

    payload = {
        "model": model_name,
        "temperature": temperature,
        "stream": False,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # server reachable but rejected the request
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ProviderError(
            f"Local model server at {base_url} returned HTTP {e.code}. "
            f"Is the model '{model_name}' installed? (e.g. 'ollama pull {model_name}'). "
            f"Server said: {detail}"
        ) from e
    except urllib.error.URLError as e:  # nothing listening / can't connect
        raise ProviderError(
            f"Could not reach a local model server at {base_url}. Start one first — "
            f"for Ollama: install it, then run 'ollama pull {model_name}' (the server "
            f"runs automatically). Or switch providers in .env: MODEL_PROVIDER=mock "
            f"(offline) or =anthropic/openai/google. (Underlying error: {e.reason})"
        ) from e
    except TimeoutError as e:  # connected, but the model took >180s to answer
        raise ProviderError(
            f"The local model '{model_name}' at {base_url} did not answer within "
            f"180 seconds. On CPU-only machines large models can be this slow — "
            f"try a smaller model, or check the server isn't stuck ('ollama ps')."
        ) from e
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(
            f"Unexpected response from the local server at {base_url}: {repr(body)[:300]}"
        ) from e


def _require_key(env_var: str) -> str:
    val = os.getenv(env_var)
    if not val:
        raise ProviderError(
            f"{env_var} is not set. Add it to your .env file before using this provider."
        )
    return val


def _openai(system: str, user: str, model: str | None, temperature: float) -> str:
    _require_key("OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ProviderError("Package 'openai' not installed. Run: pip install openai") from e
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _anthropic(system: str, user: str, model: str | None, temperature: float) -> str:
    _require_key("ANTHROPIC_API_KEY")
    try:
        from anthropic import Anthropic
    except ImportError as e:  # pragma: no cover
        raise ProviderError("Package 'anthropic' not installed. Run: pip install anthropic") from e
    client = Anthropic()
    resp = client.messages.create(
        model=model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6",
        max_tokens=1500,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


def _google(system: str, user: str, model: str | None, temperature: float) -> str:
    _require_key("GOOGLE_API_KEY")
    try:
        import google.generativeai as genai
    except ImportError as e:  # pragma: no cover
        raise ProviderError(
            "Package 'google-generativeai' not installed. Run: pip install google-generativeai"
        ) from e
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gmodel = genai.GenerativeModel(
        model or os.getenv("GOOGLE_MODEL") or "gemini-1.5-flash",
        system_instruction=system,
    )
    resp = gmodel.generate_content(
        user,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    return resp.text or ""
