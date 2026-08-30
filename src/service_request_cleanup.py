"""Project 2A — Service Request & Technical Comment Cleanup Assistant.

Takes messy support / technical notes and returns a clean, professional,
structured record (validated against ``ServiceRequestResult``).

Run from the project root:
    python cleanup.py "cust says cant login been trying since morning very upset"
    python cleanup.py --file note.txt --provider openai
    echo "some note" | python cleanup.py --json-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

from pydantic import ValidationError

from .formatters import service_request_summary, service_request_to_text
from .provider_adapter import ProviderError, complete, is_offline, resolve_provider
from .schemas import ServiceRequestResult

logger = logging.getLogger("service_request_cleanup")


class CleanupError(Exception):
    """Base error for this utility."""


class EmptyInputError(CleanupError):
    """Raised when no service-request text was provided."""


class SchemaValidationError(CleanupError):
    """Raised when the model never returned schema-valid JSON, even after retries."""


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are a service desk assistant. You clean up messy service request notes and
technical comments into a clean, professional, structured record.

You are given raw text written by an operator, support analyst, developer, or
customer. Respond with a SINGLE JSON object and nothing else — no markdown
fences, no commentary.

The JSON object MUST contain exactly these fields:
  "cleaned_description": string  - professional rewritten version of the issue
  "short_summary":       string  - one sentence
  "issue_type":          one of ["Login","Payment","Refund","Order","Loyalty","Technical","Data","Integration","Other"]
  "customer_impact":     string  - what problem the customer or user is experiencing
  "technical_details":   string[] - any system, error, file, API, screen, or process mentioned
  "missing_information": string[] - information needed to resolve the issue
  "suggested_next_step": string  - recommended next action
  "tone_adjustment":     string  - note if the original was emotional/unclear/unprofessional and how you neutralised it
  "ml_tags":             string[] - simple, consistent, reusable tags for future classification
  "confidence_score":    integer - 0-100 based on clarity and completeness of the input

Rules:
1. Do NOT invent facts or technical details.
2. Preserve the original meaning.
3. Remove emotional, blaming, or unprofessional wording.
4. Use clear, neutral, professional language.
5. If information is missing, write "Not stated" in a text field, or list it under missing_information.
6. If the issue type is unclear, use "Other" and explain why in tone_adjustment or suggested_next_step.
7. ml_tags must be simple, lowercase, and underscore_separated (e.g. "login_issue", "password_reset").
8. Base confidence_score on how clear and complete the input is. Very short or vague input gets a low score.

Return ONLY the JSON object."""


def _normalise(text: str) -> str:
    """Strip surrounding whitespace and any BOM / zero-width no-break spaces.

    (A BOM sneaks in when text is piped on Windows; ``str.strip()`` won't remove
    it, so a BOM-only input would otherwise look non-blank.)
    """
    return (text or "").replace("﻿", "").strip()


def _build_user_prompt(text: str, repair_hint: str | None = None) -> str:
    prompt = f"Service request text to clean up:\n\n{text}"
    if repair_hint:
        prompt += (
            "\n\nYour previous response could not be used. Error:\n"
            f"{repair_hint}\n"
            "Respond again with ONLY a valid JSON object matching the schema."
        )
    return prompt


# --------------------------------------------------------------------------- #
# Offline stub (used only when provider == 'mock')
# --------------------------------------------------------------------------- #

_KEYWORD_MAP = [
    ("Login", ["log in", "login", "log-in", "password", "sign in", "signin", "locked out", "2fa", "mfa"]),
    ("Refund", ["refund", "money back", "reimburse", "chargeback"]),
    ("Payment", ["payment", "pay ", "charged", "credit card", "billing", "invoice", "declined"]),
    ("Order", ["order", "delivery", "shipment", "shipping", "package", "tracking"]),
    ("Loyalty", ["points", "rewards", "loyalty", "membership", "tier"]),
    ("Integration", ["api", "integration", "webhook", "sync", "endpoint"]),
    ("Data", ["report", "export", "missing data", "record", "spreadsheet", "csv"]),
    ("Technical", ["error", "crash", "bug", "broken", "screen", "page", "load", "freeze", "500", "404"]),
]


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    head = parts[0] if parts and parts[0] else text
    return head[:200]


def _light_clean(text: str) -> str:
    t = " ".join(text.split())
    return (t[0].upper() + t[1:]) if t else t


def _offline_stub(text: str) -> dict:
    """A schema-valid, honest placeholder result for the 'mock' provider.

    This is NOT real AI cleanup — it lets the pipeline, CLI, and tests run with
    no API key. Configure a real provider for genuine analysis.
    """
    lower = text.lower()
    issue = "Other"
    for label, words in _KEYWORD_MAP:
        if any(w in lower for w in words):
            issue = label
            break
    words = len(text.split())
    confidence = 30 if words < 6 else (60 if words < 25 else 75)
    return {
        "cleaned_description": _light_clean(text),
        "short_summary": _first_sentence(text),
        "issue_type": issue,
        "customer_impact": "Not stated",
        "technical_details": [],
        "missing_information": [
            "Offline stub result — configure a real AI provider for full analysis."
        ],
        "suggested_next_step": "Review the request and gather any missing details.",
        "tone_adjustment": "Not assessed in offline mode.",
        "ml_tags": [issue.lower() + "_issue"],
        "confidence_score": confidence,
    }


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def _extract_json(raw: str) -> dict:
    """Pull the JSON object out of a model response, tolerating fences / prose."""
    if not raw or not raw.strip():
        raise ValueError("Empty response from the model.")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in the model response.")
    return json.loads(raw[start : end + 1])


def clean_service_request(
    text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = 2,
    dev_mode: bool = False,
) -> ServiceRequestResult:
    """Clean one service request. Returns a validated ``ServiceRequestResult``.

    Raises ``EmptyInputError`` on blank input and ``SchemaValidationError`` if
    the model never produces schema-valid JSON.
    """
    text = _normalise(text)
    if not text:
        raise EmptyInputError("No service request text was provided.")

    # Privacy: customer text is only logged when dev mode is explicitly on.
    if dev_mode:
        logger.info("Input text: %s", text)

    user_prompt = _build_user_prompt(text)
    offline = _offline_stub(text)
    last_error: Exception | None = None

    # Constrain the model to our schema: guarantees valid JSON and that issue_type
    # is one of the allowed enum values (kills the invalid-JSON / out-of-enum class
    # of failures on small local models). The model still has to pick the RIGHT
    # value — that's why model quality, not just this schema, matters.
    schema = ServiceRequestResult.model_json_schema()

    for _ in range(max_retries + 1):
        raw = complete(
            _SYSTEM_PROMPT,
            user_prompt,
            provider=provider,
            model=model,
            mock_response=offline,
            response_schema=schema,
        )
        try:
            data = _extract_json(raw)
            return ServiceRequestResult.model_validate(data)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            user_prompt = _build_user_prompt(text, repair_hint=str(exc))

    raise SchemaValidationError(
        f"The model did not return valid output after {max_retries + 1} attempts: {last_error}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _read_input(args) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            return fh.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanup",
        description="Project 2A — clean up messy service request / technical notes into structured JSON.",
    )
    parser.add_argument("text", nargs="?", help="the note to clean (or use --file / stdin)")
    parser.add_argument("-f", "--file", help="read the note from a text file")
    parser.add_argument("-p", "--provider", help="local | mock | openai | anthropic | google (default: from .env)")
    parser.add_argument("-m", "--model", help="override the model name for the provider")
    parser.add_argument("--json-only", action="store_true", help="print JSON only (no human summary)")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print ONLY the one-line summary (clean output for a UI, no JSON/debug)",
    )
    parser.add_argument("--dev-mode", action="store_true", help="enable logging of input text (local dev only)")
    args = parser.parse_args(argv)

    try:  # load .env if python-dotenv is available; harmless if not
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.dev_mode:
        logging.basicConfig(level=logging.INFO)

    text = _read_input(args)

    if not _normalise(text):
        # Spec test case 6: blank input -> clear error (JSON first).
        print(json.dumps({"error": "No service request text was provided."}, indent=2))
        print("\nError: please pass a note as an argument, with --file, or via stdin.", file=sys.stderr)
        return 2

    provider_name = resolve_provider(args.provider)
    if is_offline(provider_name):
        print(
            "[note] Running with the OFFLINE 'mock' provider — results are a placeholder, "
            "not real AI analysis. Set MODEL_PROVIDER and an API key in .env for real output.",
            file=sys.stderr,
        )

    # Per-tool model: 2A can run a different (smaller/faster) model than 3A.
    # Priority: --model flag > CLEANUP_MODEL env > LOCAL_MODEL env (resolved later).
    chosen_model = args.model or os.getenv("CLEANUP_MODEL")

    try:
        result = clean_service_request(
            text, provider=args.provider, model=chosen_model, dev_mode=args.dev_mode
        )
    except ProviderError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    except CleanupError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    # UI-facing mode: just the clean one-line summary, nothing else.
    if args.summary_only:
        print(service_request_summary(result))
        return 0

    # Machine-readable JSON first, human-readable summary second (per spec).
    print(result.model_dump_json(indent=2))
    if not args.json_only:
        print()
        print(service_request_to_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
