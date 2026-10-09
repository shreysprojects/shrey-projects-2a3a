"""Project 3A — Translation quality EVALUATION harness (human-in-the-loop).

Why this exists
---------------
Automated checks alone do not prove the translation reviewer is reliable enough
to trust on customer-facing messages. Passing 10 hand-picked sentences tells you
very little about the long tail (negation drops, wrong numbers, added promises,
subtle tone shifts). This harness supports a two-stage validation Ravi asked for:

  1) AUTOMATED PASS  (--run)
     Run the checker over a labelled eval set and measure how often its
     send_recommendation matches the human-curated expected label. Reports overall
     agreement AND the two failure modes that actually matter for the business:
       * SAFETY MISSES  — expected "Do Not Send" but the model would have sent /
         only flagged for review. These are the dangerous ones.
       * FALSE ALARMS   — expected "Send" but the model blocked it. These annoy
         operators and erode trust.
     It also writes a REVIEW SHEET (CSV) with blank columns for a reviewer.

  2) HUMAN PASS      (--score reviewed.csv)
     After a professional / native bilingual reviewer fills in their own verdict
     in the sheet, re-import it to measure model-vs-human and human-vs-expected
     agreement and to list every disagreement for discussion.

Runs against whatever provider MODEL_PROVIDER / --provider selects (mock, openai,
anthropic, google, or a future self-hosted local model — see
docs/local-llm-shortlist.md). Standard library only; no pandas.

Run from the project root:
    python samples/evaluation/evaluate_translations.py --run --provider openai
    python samples/evaluation/evaluate_translations.py --score samples/evaluation/review_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:  # load .env so MODEL_PROVIDER / API keys are picked up (same as the CLI)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.provider_adapter import ProviderError, is_offline, resolve_provider  # noqa: E402
from src.translation_quality_checker import (  # noqa: E402
    EmptyInputError,
    NotSpanishError,
    TranslationError,
    check_translation,
)

HERE = pathlib.Path(__file__).resolve().parent
EVAL_SET = HERE / "translation_eval_set.json"
DEFAULT_SHEET = HERE / "review_sheet.csv"

# The columns the tool fills, then the three the human reviewer fills.
MODEL_COLUMNS = [
    "id", "category", "english_original", "spanish_translation",
    "expected_recommendation", "model_recommendation", "model_matches_expected",
    "model_accuracy", "model_confidence", "model_short_summary",
    "model_back_translation", "model_suggested_correction", "model_risky_phrases",
]
HUMAN_COLUMNS = ["human_recommendation", "human_agrees_with_model", "reviewer_notes"]
ALL_COLUMNS = MODEL_COLUMNS + HUMAN_COLUMNS

VALID_RECS = {"Send", "Review First", "Do Not Send", "Error"}


def _display_path(p: pathlib.Path) -> str:
    """Path relative to the project root for a copy-pasteable hint, falling back
    to the path as given when it lives outside the root (e.g. an absolute --out
    somewhere else). Avoids relative_to() raising on unrelated paths."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _norm_rec(value: str) -> str:
    """Normalise a recommendation label typed by a human (casing/spacing)."""
    cleaned = " ".join((value or "").split()).strip().title()
    # ".Title()" turns "Do Not Send" fine; map a couple of common variants.
    aliases = {"Dont Send": "Do Not Send", "Do Not Send": "Do Not Send",
               "Review": "Review First", "Hold": "Review First"}
    return aliases.get(cleaned, cleaned)


# --------------------------------------------------------------------------- #
# Stage 1 — automated pass + review-sheet export
# --------------------------------------------------------------------------- #

def _run_one(case: dict, provider: str | None, model: str | None) -> dict:
    """Run the checker on one case and return a flat row for the CSV."""
    row = {
        "id": case["id"],
        "category": case.get("category", ""),
        "english_original": case["english_original"],
        "spanish_translation": case["spanish_translation"],
        "expected_recommendation": case.get("expected_recommendation", ""),
        "model_recommendation": "",
        "model_matches_expected": "",
        "model_accuracy": "",
        "model_confidence": "",
        "model_short_summary": "",
        "model_back_translation": "",
        "model_suggested_correction": "",
        "model_risky_phrases": "",
        # human columns start blank
        "human_recommendation": "",
        "human_agrees_with_model": "",
        "reviewer_notes": "",
    }
    try:
        result = check_translation(
            case["english_original"], case["spanish_translation"],
            provider=provider, model=model,
        )
        row["model_recommendation"] = result.send_recommendation.value
        row["model_accuracy"] = result.accuracy_rating.value
        row["model_confidence"] = result.confidence_score
        row["model_short_summary"] = result.short_summary
        row["model_back_translation"] = result.back_translation
        row["model_suggested_correction"] = result.suggested_correction
        row["model_risky_phrases"] = " | ".join(result.risky_phrases)
    except (NotSpanishError, EmptyInputError) as exc:
        # A rejection IS a valid outcome — the input guard fired before the model.
        row["model_recommendation"] = "Error"
        row["model_short_summary"] = f"Rejected by input guard: {exc}"
    except (ProviderError, TranslationError) as exc:
        row["model_recommendation"] = "Error"
        row["model_short_summary"] = f"Run error: {exc}"

    expected = row["expected_recommendation"]
    row["model_matches_expected"] = (
        "yes" if expected and row["model_recommendation"] == expected else "no"
    )
    return row


def run(provider: str | None, model: str | None, out_path: pathlib.Path) -> int:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    resolved = resolve_provider(provider)
    print(f"Running {len(cases)} cases against provider '{resolved}'...")
    if is_offline(resolved):
        print(
            "[note] Provider is the OFFLINE 'mock' — it cannot judge translation "
            "quality, so agreement numbers below are NOT meaningful. Configure a "
            "real or local provider to get a real evaluation.\n",
            file=sys.stderr,
        )

    rows = [_run_one(c, provider, model) for c in cases]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    _print_automated_report(rows)
    print(f"\nReview sheet written to: {out_path}")
    print(
        "Next: have a bilingual reviewer fill in 'human_recommendation' "
        "(Send / Review First / Do Not Send), then run:\n"
        f"  python {_display_path(pathlib.Path(__file__))} --score {_display_path(out_path)}"
    )
    return 0


def _print_automated_report(rows: list[dict]) -> None:
    total = len(rows)
    scored = [r for r in rows if r["expected_recommendation"]]
    matches = [r for r in scored if r["model_matches_expected"] == "yes"]

    # Safety miss: expected "Do Not Send" but model would not block it.
    safety_misses = [
        r for r in scored
        if r["expected_recommendation"] == "Do Not Send"
        and r["model_recommendation"] != "Do Not Send"
    ]
    # False alarm: expected "Send" but model blocked it outright.
    false_alarms = [
        r for r in scored
        if r["expected_recommendation"] == "Send"
        and r["model_recommendation"] == "Do Not Send"
    ]

    print("\n" + "=" * 60)
    print("AUTOMATED PASS — model vs. expected label")
    print("=" * 60)
    pct = (len(matches) / len(scored) * 100) if scored else 0.0
    print(f"Overall agreement : {len(matches)}/{len(scored)}  ({pct:.0f}%)")
    print(f"Safety misses     : {len(safety_misses)}   (expected 'Do Not Send', model did not block)")
    print(f"False alarms      : {len(false_alarms)}   (expected 'Send', model blocked)")

    disagreements = [r for r in scored if r["model_matches_expected"] == "no"]
    if disagreements:
        print("\nDisagreements to inspect:")
        for r in disagreements:
            flag = "  [SAFETY MISS]" if r in safety_misses else ""
            print(
                f"  #{r['id']:>2} {r['category']:<22} "
                f"expected={r['expected_recommendation']:<12} "
                f"model={r['model_recommendation']}{flag}"
            )


# --------------------------------------------------------------------------- #
# Stage 2 — score a reviewed sheet (human-in-the-loop)
# --------------------------------------------------------------------------- #

def score(sheet_path: pathlib.Path) -> int:
    if not sheet_path.exists():
        print(f"Error: review sheet not found: {sheet_path}", file=sys.stderr)
        return 2
    with sheet_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("Error: the review sheet is empty.", file=sys.stderr)
        return 2

    reviewed = []
    unknown = []
    for r in rows:
        human = _norm_rec(r.get("human_recommendation", ""))
        if not human:
            continue
        if human not in VALID_RECS:
            unknown.append((r.get("id", "?"), r.get("human_recommendation", "")))
            continue
        r["_human"] = human
        reviewed.append(r)

    total = len(rows)
    print("=" * 60)
    print("HUMAN PASS — reviewed sheet")
    print("=" * 60)
    print(f"Cases in sheet         : {total}")
    print(f"Reviewed by a human    : {len(reviewed)}")
    if unknown:
        print(f"Unrecognised verdicts  : {len(unknown)} -> {unknown}")
    if not reviewed:
        print(
            "\nNo human verdicts filled in yet. Add 'human_recommendation' values "
            "(Send / Review First / Do Not Send) and re-run --score."
        )
        return 0

    hm = [r for r in reviewed if r["_human"] == r.get("model_recommendation", "")]
    he = [
        r for r in reviewed
        if r.get("expected_recommendation") and r["_human"] == r["expected_recommendation"]
    ]
    hm_pct = len(hm) / len(reviewed) * 100
    he_scored = [r for r in reviewed if r.get("expected_recommendation")]
    he_pct = (len(he) / len(he_scored) * 100) if he_scored else 0.0

    print(f"\nHuman vs. model        : {len(hm)}/{len(reviewed)}  ({hm_pct:.0f}%)")
    print(f"Human vs. expected set : {len(he)}/{len(he_scored)}  ({he_pct:.0f}%)")

    # Where the human overruled the model (the real signal for reliability).
    overrides = [r for r in reviewed if r["_human"] != r.get("model_recommendation", "")]
    if overrides:
        print("\nHuman overruled the model on:")
        for r in overrides:
            print(
                f"  #{r.get('id','?'):>2} {r.get('category',''):<22} "
                f"model={r.get('model_recommendation',''):<12} "
                f"human={r['_human']:<12} "
                f"note={r.get('reviewer_notes','').strip() or '-'}"
            )
    else:
        print("\nThe human agreed with the model on every reviewed case.")

    print(
        "\nGuidance: for a provider/model to be trusted on customer-facing sends, "
        "aim for high human-vs-model agreement AND zero human-flagged safety misses "
        "(a 'Do Not Send' the model would have sent). See samples/evaluation/README.md."
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate_translations",
        description="Human-in-the-loop evaluation harness for Project 3A.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="run the automated pass and write a review sheet (default)")
    mode.add_argument("--score", metavar="CSV", help="score a reviewed sheet filled in by a human")
    parser.add_argument("-p", "--provider", help="local | mock | openai | anthropic | google (default: from .env)")
    parser.add_argument("-m", "--model", help="override the model name for the provider")
    parser.add_argument("-o", "--out", help=f"where to write the review sheet (default: {DEFAULT_SHEET})")
    args = parser.parse_args(argv)

    if args.score:
        return score(pathlib.Path(args.score))
    out_path = pathlib.Path(args.out) if args.out else DEFAULT_SHEET
    return run(args.provider, args.model, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
