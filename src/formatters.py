"""Human-readable formatters.

The spec requires both machine-readable JSON *and* a human-readable summary.
The JSON comes straight from the Pydantic model; these helpers produce the
readable block.
"""
from __future__ import annotations

from .schemas import ServiceRequestResult, TranslationQualityResult


def _bullet_list(items, empty="(none)") -> str:
    if not items:
        return f"  {empty}"
    return "\n".join(f"  - {item}" for item in items)


def service_request_summary(result: ServiceRequestResult) -> str:
    """The single clean line for the UI (Project 2A) — no JSON, no debug detail."""
    return result.short_summary.strip()


def translation_quality_summary(result: TranslationQualityResult) -> str:
    """The single clean line for the UI (Project 3A) — no JSON, no debug detail."""
    return result.short_summary.strip()


def service_request_to_text(result: ServiceRequestResult) -> str:
    """Render a ServiceRequestResult (Project 2A) as a readable block."""
    return "\n".join(
        [
            "SERVICE REQUEST — CLEANED",
            "=" * 42,
            f"Summary      : {result.short_summary}",
            f"Issue type   : {result.issue_type.value}",
            f"Confidence   : {result.confidence_score}/100",
            "",
            "Cleaned description:",
            f"  {result.cleaned_description}",
            "",
            f"Customer impact:",
            f"  {result.customer_impact}",
            "",
            "Technical details:",
            _bullet_list(result.technical_details),
            "",
            "Missing information:",
            _bullet_list(result.missing_information),
            "",
            f"Suggested next step:",
            f"  {result.suggested_next_step}",
            "",
            f"Tone adjustment:",
            f"  {result.tone_adjustment}",
            "",
            f"ML tags: {', '.join(result.ml_tags) if result.ml_tags else '(none)'}",
        ]
    )


def translation_quality_to_text(result: TranslationQualityResult) -> str:
    """Render a TranslationQualityResult (Project 3A) as a readable block."""
    lines = [
        "TRANSLATION QUALITY CHECK",
        "=" * 42,
        f"Summary        : {result.short_summary}",
        f"Recommendation : {result.send_recommendation.value}",
        f"Accuracy       : {result.accuracy_rating.value}",
        f"Tone           : {result.tone_check.value}",
        f"Confidence     : {result.confidence_score}/100",
        "",
        "Risky phrases:",
        _bullet_list(result.risky_phrases),
        "",
        "Missing meaning (in English, absent from Spanish):",
        _bullet_list(result.missing_meaning),
        "",
        "Added meaning (in Spanish, not in English):",
        _bullet_list(result.added_meaning),
        "",
        "Back-translation (Spanish -> English, for review):",
        f"  {result.back_translation or '(none)'}",
    ]
    if result.suggested_correction.strip():
        lines += [
            "",
            "Suggested corrected Spanish:",
            f"  {result.suggested_correction}",
        ]
    lines += [
        "",
        "Explanation:",
        f"  {result.explanation or '(none)'}",
    ]
    return "\n".join(lines)
