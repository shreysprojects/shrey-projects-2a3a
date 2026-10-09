"""Project 3A — optional semantic (NLI) check layer.

A small open-source cross-lingual model (mDeBERTa-v3 XNLI, MIT licence, ~280M
params, runs locally on CPU in under a second per pair) that directly compares
the English and the Spanish and asks two questions:

  * does the Spanish CONTRADICT the English?      -> floor "Do Not Send"
  * does either side fail to FOLLOW from the other -> floor "Review First"
    (content added or dropped)

This catches the error class no word-rule can: pure meaning swaps with no
trigger vocabulary ("your subscription is active" -> "ha sido suspendida").
Measured on the 52-case eval set it raised agreement from 63% to 80% with zero
false alarms (2026-07-08).

Like the deterministic safety gate, this layer is ESCALATE-ONLY: it can make
the verdict stricter, never approve anything.

OPTIONAL BY DESIGN. Needs the heavy ML stack (pip install -r
requirements-nli.txt, ~2.5GB; first run downloads the model, ~1.1GB). When the
stack is missing — or NLI_CHECK=off in the environment — every function here
is a silent no-op and 3A runs exactly as before (model + safety gate). The
model loads once per process (~10s): fine for the web UI / server, which
preloads it in the background at startup; a one-shot CLI run pays that load
each time, so the UI is the recommended surface when this layer matters.

Thresholds are provisional: they sit in a wide measured margin (flagged cases
scored <=0.09 entailment vs >=0.51 for clean ones) but should be re-validated
as the eval set grows.
"""
from __future__ import annotations

import os
import sys

from .safety_gate import GateFinding, escalate
from .schemas import SendRecommendation, TranslationQualityResult

_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# P(contradiction) above this in BOTH directions -> Do Not Send floor.
# Measured on the 152-case set (2026-07-08): genuine meaning reversals score
# ~1.00 in both directions; clean pairs that spuriously trigger one direction
# never exceeded 0.78 in the weaker one. Requiring agreement kills those
# false blocks while keeping every true reversal.
CONTRADICTION_BLOCK = 0.85
# P(entailment) below this (either direction) -> Review First floor.
ENTAILMENT_REVIEW = 0.20

# Lazy singleton: (tokenizer, model), or "failed" after an unrecoverable error.
_state: dict = {"pipe": None, "failed": False}


def is_enabled() -> bool:
    """True when the layer is switched on AND the ML stack is importable."""
    if (os.getenv("NLI_CHECK") or "on").strip().lower() in ("off", "0", "false", "no"):
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_pipe():
    """Load the model once per process. Returns None if unavailable."""
    if _state["failed"] or not is_enabled():
        return None
    if _state["pipe"] is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
            mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            mdl.eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
            _state["pipe"] = (tok, mdl)
        except Exception as exc:  # e.g. no network for the first download
            _state["failed"] = True
            print(
                f"[nli_check] semantic layer disabled ({type(exc).__name__}: {exc}). "
                "3A continues with model + safety gate only.",
                file=sys.stderr,
            )
            return None
    return _state["pipe"]


def warm_up() -> bool:
    """Preload the model (used by the web UI at startup). True if ready."""
    return _get_pipe() is not None


def _scores(premise: str, hypothesis: str) -> dict:
    """{'entailment': p, 'neutral': p, 'contradiction': p} for one direction."""
    import torch

    tok, mdl = _state["pipe"]
    inputs = tok(premise, hypothesis, truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(mdl(**inputs).logits[0], dim=-1).tolist()
    return {mdl.config.id2label[i].lower(): p for i, p in enumerate(probs)}


def run_nli_check(english: str, spanish: str) -> list[GateFinding]:
    """Bidirectional NLI on the pair. Empty list when clean OR unavailable."""
    if _get_pipe() is None:
        return []
    try:
        fwd = _scores(english, spanish)
        bwd = _scores(spanish, english)
    except Exception as exc:  # never let this layer break the pipeline
        print(f"[nli_check] scoring failed, skipping: {exc}", file=sys.stderr)
        return []

    # min(): both directions must independently see the contradiction.
    contradiction = min(fwd["contradiction"], bwd["contradiction"])
    weakest_entailment = min(fwd["entailment"], bwd["entailment"])

    if contradiction > CONTRADICTION_BLOCK:
        return [GateFinding(
            f"the Spanish and English contradict each other in both reading "
            f"directions (confidence {contradiction:.0%}) — the meaning may be changed or reversed",
            SendRecommendation.DO_NOT_SEND,
        )]
    if weakest_entailment < ENTAILMENT_REVIEW:
        return [GateFinding(
            f"one side says something the other does not support "
            f"(agreement {weakest_entailment:.0%}) — content may be added or missing",
            SendRecommendation.REVIEW_FIRST,
        )]
    return []


def apply_nli_check(
    result: TranslationQualityResult, english: str, spanish: str
) -> TranslationQualityResult:
    """Escalate the recommendation if the semantic check found a problem."""
    return escalate(result, run_nli_check(english, spanish), "semantic check")
