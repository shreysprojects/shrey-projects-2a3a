"""Project 3A — Translation Quality Checker.

Compares an English customer-service message with its Spanish translation and
returns a structured, validated quality assessment: accuracy, tone, risky
phrases, missing/added meaning, an optional improved Spanish version, a
back-translation for the operator, and a send/hold/block recommendation.

Run from the project root:
    python check_translation.py "English text" "Spanish text"
    python check_translation.py --english "..." --spanish "..."
    python check_translation.py --file pair.json --provider openai
    echo '{"english_original":"...","spanish_translation":"..."}' | python check_translation.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

from pydantic import ValidationError

from .formatters import translation_quality_summary, translation_quality_to_text
from .nli_check import apply_nli_check
from .provider_adapter import ProviderError, complete, is_offline, resolve_provider
from .safety_gate import apply_safety_gate
from .schemas import TranslationQualityResult

logger = logging.getLogger("translation_quality_checker")


class TranslationError(Exception):
    """Base error for this utility."""


class EmptyInputError(TranslationError):
    """Raised when the English or Spanish text is blank."""


class NotSpanishError(TranslationError):
    """Raised when the 'Spanish' text does not appear to be Spanish (spec rule 15)."""


class SchemaValidationError(TranslationError):
    """Raised when the model never returned schema-valid JSON, even after retries."""


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are a bilingual (English/Spanish) customer-service quality reviewer. You are
given an original English message and its Spanish translation. You judge whether
the Spanish is accurate, professional, tone-appropriate, and safe to send to a
customer.

Respond with a SINGLE JSON object and nothing else — no markdown fences, no
commentary.

The JSON object MUST contain exactly these fields:
  "short_summary":        string    - ONE short plain-English sentence a busy operator can read at a glance: the verdict and the single main reason (e.g. "Safe to send: accurate and professional." or "Do not send: adds an unauthorised refund promise not in the English.")
  "accuracy_rating":      one of ["High","Medium","Low"]  - how faithfully the Spanish conveys the English
  "confidence_score":     integer 0-100  - your confidence in this assessment
  "tone_check":           one of ["Friendly","Professional","Too Harsh","Too Casual","Unclear"]
  "risky_phrases":        string[]  - wording that could confuse, offend, or create business risk
  "missing_meaning":      string[]  - meaning present in English but absent from the Spanish
  "added_meaning":        string[]  - meaning present in the Spanish but NOT in the English
  "suggested_correction": string    - an improved Spanish version ONLY if a correction is needed, else ""
  "back_translation":     string    - the Spanish translated literally back into English, for operator review
  "send_recommendation":  one of ["Send","Review First","Do Not Send"]
  "explanation":          string    - one or two sentences justifying the recommendation

Rules:
1. Do NOT rewrite the translation unless a correction is actually needed. If it is fine, leave suggested_correction as "".
2. Do NOT make the Spanish more legally binding or more of a promise than the English original.
3. Preserve a polite customer-service tone.
4. Flag rude, harsh, overly direct, or confusing language in risky_phrases and reflect it in tone_check.
5. Flag added order numbers, refund promises, legal claims, guarantees, or commitments that are NOT in the English original — these go in added_meaning and usually mean "Do Not Send".
6. If meaning from the English is dropped, list it in missing_meaning (usually "Review First").
7. Choose send_recommendation by what is actually wrong, not by overall strictness:
   - "Send" only when accuracy is High, tone is appropriate, and nothing risky was added or dropped.
   - "Review First" when the meaning is essentially preserved but the wording needs a human
     look first — e.g. tone is too harsh or too casual, or some detail is missing. A tone problem
     ALONE (harsh/casual) is "Review First", not "Do Not Send"; put the fix in suggested_correction.
   - "Do Not Send" ONLY when the meaning is wrong, contradictory, or reversed, OR an unauthorised
     promise / refund / guarantee / legal claim / commitment was added that is not in the English.
8. The back_translation must reflect what the Spanish ACTUALLY says, even if that differs from the English.
9. short_summary must agree with send_recommendation and stay under ~20 words. It is the line the UI shows; keep it clean and non-technical (no jargon, no JSON, no field names).

Return ONLY the JSON object."""


def _build_user_prompt(english: str, spanish: str, repair_hint: str | None = None) -> str:
    prompt = (
        "Original English message:\n"
        f"{english}\n\n"
        "Spanish translation to review:\n"
        f"{spanish}"
    )
    if repair_hint:
        prompt += (
            "\n\nYour previous response could not be used. Error:\n"
            f"{repair_hint}\n"
            "Respond again with ONLY a valid JSON object matching the schema."
        )
    return prompt


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    """Strip surrounding whitespace plus any BOM that sneaks in via stdin/pipe."""
    return (text or "").replace("﻿", "").strip()


# Signals used only as a lightweight "is this Spanish?" guard (spec rule 15).
# This is intentionally conservative: any Spanish-specific character or common
# Spanish word counts as Spanish, so genuine Spanish is never rejected.
_SPANISH_CHARS = re.compile(r"[ñáéíóúü¿¡]", re.IGNORECASE)
_SPANISH_WORDS = {
    "que", "de", "la", "el", "los", "las", "un", "una", "su", "para", "no",
    "se", "es", "con", "por", "y", "en", "le", "lo", "del", "al", "como",
    "mas", "pero", "este", "esta", "esto", "envie", "envienos", "gracias",
    "disculpe", "disculpa", "lamentamos", "pedido", "numero", "problema",
    "ayuda", "cuenta", "correo", "nosotros", "usted", "puede", "sentimos",
    "revisar", "haya", "retrasado", "lo", "sus", "nos", "muy",
}
_ENGLISH_WORDS = {
    "the", "your", "you", "please", "we", "are", "sorry", "order", "number",
    "so", "can", "review", "issue", "is", "was", "and", "to", "of", "for",
    "with", "this", "that", "help", "account", "email", "not", "working",
    "send", "us", "our", "will", "have", "been", "delayed",
}


def _looks_spanish(text: str) -> bool:
    """Best-effort check that ``text`` is Spanish, not English or empty.

    Returns True if there is any Spanish signal. Returns False only when the
    text shows clear English signals and no Spanish ones (the common mistake of
    pasting the English message into the Spanish field).
    """
    if _SPANISH_CHARS.search(text):
        return True
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    spanish_hits = sum(1 for w in words if w in _SPANISH_WORDS)
    english_hits = sum(1 for w in words if w in _ENGLISH_WORDS)
    if spanish_hits:
        return True
    # No Spanish signal at all: treat as non-Spanish only if it clearly reads
    # as English. Otherwise (e.g. a proper noun, a number) give it the benefit
    # of the doubt and let the model judge.
    return english_hits < 2


# --------------------------------------------------------------------------- #
# Offline stub (used only when provider == 'mock')
# --------------------------------------------------------------------------- #

def _offline_stub(english: str, spanish: str) -> dict:
    """A schema-valid, honest placeholder for the 'mock' provider.

    This is NOT a real bilingual review — it lets the pipeline, CLI, and tests
    run with no API key. It never recommends "Send" on its own (it can't judge
    accuracy), so the safe default is "Review First". Configure a real provider
    for a genuine assessment.
    """
    return {
        "short_summary": "Review first — offline mock mode did not assess this translation.",
        "accuracy_rating": "Medium",
        "confidence_score": 30,
        "tone_check": "Unclear",
        "risky_phrases": [],
        "missing_meaning": [],
        "added_meaning": [],
        "suggested_correction": "",
        "back_translation": "Not produced in offline mode.",
        "send_recommendation": "Review First",
        "explanation": (
            "Offline stub result — no real bilingual analysis was performed. "
            "Configure a real AI provider (openai/anthropic/google) for an "
            "accurate assessment."
        ),
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


def check_translation(
    english: str,
    spanish: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = 2,
    dev_mode: bool = False,
) -> TranslationQualityResult:
    """Check one English -> Spanish pair. Returns a validated result.

    Raises ``EmptyInputError`` if either side is blank, ``NotSpanishError`` if
    the Spanish text doesn't look like Spanish, and ``SchemaValidationError`` if
    the model never produces schema-valid JSON.
    """
    english = _normalise(english)
    spanish = _normalise(spanish)
    if not english:
        raise EmptyInputError("No English original was provided.")
    if not spanish:
        raise EmptyInputError("No Spanish translation was provided.")
    if not _looks_spanish(spanish):
        raise NotSpanishError(
            "The 'Spanish translation' does not appear to be Spanish. "
            "Please provide the Spanish text."
        )

    # Privacy: customer text is only logged when dev mode is explicitly on.
    if dev_mode:
        logger.info("English: %s", english)
        logger.info("Spanish: %s", spanish)

    user_prompt = _build_user_prompt(english, spanish)
    offline = _offline_stub(english, spanish)
    last_error: Exception | None = None

    # Constrain local-provider output to the schema (guaranteed-valid JSON with
    # in-enum values — same mechanism 2A uses; matters for small local models).
    schema = TranslationQualityResult.model_json_schema()

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
            result = TranslationQualityResult.model_validate(data)
            # Two escalate-only nets, run on every path (any provider/model):
            # 1. deterministic safety gate (rules: numbers, commitments, negation)
            # 2. semantic NLI check (optional; silent no-op if not installed)
            result = apply_safety_gate(result, english, spanish)
            return apply_nli_check(result, english, spanish)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            user_prompt = _build_user_prompt(english, spanish, repair_hint=str(exc))

    raise SchemaValidationError(
        f"The model did not return valid output after {max_retries + 1} attempts: {last_error}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _read_pair(args) -> tuple[str, str]:
    """Resolve the (english, spanish) pair from args / --file / stdin.

    Accepted shapes:
      * two positional args:           check_translation.py "EN" "ES"
      * flags:                         --english "EN" --spanish "ES"
      * a JSON file or stdin object:   {"english_original": "...", "spanish_translation": "..."}
    """
    if args.english is not None or args.spanish is not None:
        return args.english or "", args.spanish or ""
    if args.english_pos is not None or args.spanish_pos is not None:
        return args.english_pos or "", args.spanish_pos or ""

    raw = None
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            raw = fh.read()
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()

    if raw and raw.strip():
        data = json.loads(raw)
        return (
            data.get("english_original", "") or data.get("english", ""),
            data.get("spanish_translation", "") or data.get("spanish", ""),
        )
    return "", ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_translation",
        description="Project 3A — check whether an English->Spanish customer message is accurate and safe to send.",
    )
    parser.add_argument("english_pos", nargs="?", help="the English original (positional)")
    parser.add_argument("spanish_pos", nargs="?", help="the Spanish translation (positional)")
    parser.add_argument("-e", "--english", help="the English original")
    parser.add_argument("-s", "--spanish", help="the Spanish translation")
    parser.add_argument("-f", "--file", help='JSON file with {"english_original":..., "spanish_translation":...}')
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

    try:
        english, spanish = _read_pair(args)
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": f"Could not read input: {exc}"}, indent=2))
        return 2

    if not _normalise(english) or not _normalise(spanish):
        # Spec rule 15 / test case 5: blank input -> clear error (JSON first).
        msg = "Both an English original and a Spanish translation are required."
        print(json.dumps({"error": msg}, indent=2))
        print(
            "\nError: pass both texts, e.g. "
            'check_translation.py "English..." "Spanish..."',
            file=sys.stderr,
        )
        return 2

    provider_name = resolve_provider(args.provider)
    if is_offline(provider_name):
        print(
            "[note] Running with the OFFLINE 'mock' provider — results are a placeholder, "
            "not real translation analysis. Set MODEL_PROVIDER and an API key in .env for real output.",
            file=sys.stderr,
        )

    # Per-tool model: 3A can pin its own model independently of 2A.
    # Priority: --model flag > TRANSLATION_MODEL env > LOCAL_MODEL env (resolved later).
    chosen_model = args.model or os.getenv("TRANSLATION_MODEL")

    try:
        result = check_translation(
            english, spanish, provider=args.provider, model=chosen_model, dev_mode=args.dev_mode
        )
    except ProviderError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    except NotSpanishError as exc:
        # Clear, structured error result (spec rule 15).
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    except TranslationError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    # UI-facing mode: just the clean one-line summary, nothing else.
    if args.summary_only:
        print(translation_quality_summary(result))
        return 0

    # Machine-readable JSON first, human-readable summary second (per spec).
    print(result.model_dump_json(indent=2))
    if not args.json_only:
        print()
        print(translation_quality_to_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
