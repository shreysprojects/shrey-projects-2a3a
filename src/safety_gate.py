"""Project 3A — deterministic safety gate.

A rule layer that runs AFTER the AI model has judged a translation. It re-checks
the English/Spanish pair for the specific error classes that are too dangerous
to leave to a model alone — and that small, CPU-fast models demonstrably miss
(see samples/evaluation/): unauthorised commitments, changed numbers, dropped
negations.

Design rules:
  * ESCALATE-ONLY. The gate can make the recommendation stricter
    (Send -> Review First -> Do Not Send); it can NEVER loosen it. A clean gate
    pass proves nothing — the model's judgement still stands.
  * DETERMINISTIC. Pure stdlib string/regex checks: same input, same output,
    ~zero latency, works identically on any machine (no model, no download).
  * PRECISION over recall. Each rule fires only on a specific high-risk pattern
    with no licence for it in the English. The model remains responsible for
    everything the rules don't cover (tone, dropped conditions, fluency).

The commitment lexicon encodes ERROR CLASSES (guarantee/refund/deadline/legal
vocabulary), not the sentences of our eval set — new phrasings of the same
classes should still be caught.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import SendRecommendation, TranslationQualityResult

# Ordering used to compare recommendation strictness.
_STRICTNESS = {
    SendRecommendation.SEND: 0,
    SendRecommendation.REVIEW_FIRST: 1,
    SendRecommendation.DO_NOT_SEND: 2,
}


@dataclass(frozen=True)
class GateFinding:
    """One rule hit: what was found and how severe it is."""

    reason: str
    severity: SendRecommendation


# --------------------------------------------------------------------------- #
# Numbers: amounts, counts, times. A number that differs between the two texts
# is a factual error (wrong refund amount, wrong date) — never a style choice.
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _to_value(token: str) -> float:
    """Parse '45', '45.50', '45,50', '1,000' into a comparable float.

    Handles the English/Spanish separator swap: if both separators appear, the
    LAST one is the decimal mark; a single separator followed by exactly three
    digits is treated as a thousands separator ('1,000' == '1.000' == 1000).
    """
    if "." in token and "," in token:
        dec = max(token.rfind("."), token.rfind(","))
        digits = re.sub(r"[.,]", "", token[:dec])
        return float(f"{digits}.{token[dec + 1:]}")
    for sep in (".", ","):
        if sep in token:
            head, _, tail = token.rpartition(sep)
            if len(tail) == 3:  # thousands grouping
                return float(token.replace(sep, ""))
            return float(f"{head.replace(sep, '')}.{tail}")
    return float(token)


def _number_values(text: str) -> set[float]:
    return {_to_value(m.group()) for m in _NUM_RE.finditer(text)}


# Currency markers: an amount that keeps its digits but changes currency is
# still a factual error ($100 refunded != 100 € refunded).
_CURRENCY_MARKERS: list[tuple[str, re.Pattern]] = [
    ("dollars", re.compile(r"\$|\busd\b|d[oó]lar", re.IGNORECASE)),
    ("euros",   re.compile(r"€|\beur\b|\beuros?\b", re.IGNORECASE)),
    ("pounds",  re.compile(r"£|\bgbp\b|\blibras?\b", re.IGNORECASE)),
]


def _currencies(text: str) -> set[str]:
    return {name for name, rx in _CURRENCY_MARKERS if rx.search(text)}


def _check_numbers(english: str, spanish: str) -> list[GateFinding]:
    en_nums, es_nums = _number_values(english), _number_values(spanish)
    findings = []
    added_currency = _currencies(spanish) - _currencies(english)
    if added_currency:
        findings.append(GateFinding(
            f"currency ({', '.join(sorted(added_currency))}) appears in the Spanish "
            "but not in the English — the currency may have been changed",
            SendRecommendation.DO_NOT_SEND,
        ))
    added = es_nums - en_nums
    missing = en_nums - es_nums
    if added:
        pretty = ", ".join(_fmt(n) for n in sorted(added))
        findings.append(GateFinding(
            f"number(s) {pretty} appear in the Spanish but not in the English",
            SendRecommendation.DO_NOT_SEND,
        ))
    if missing:
        pretty = ", ".join(_fmt(n) for n in sorted(missing))
        findings.append(GateFinding(
            f"number(s) {pretty} from the English are missing in the Spanish",
            SendRecommendation.REVIEW_FIRST,
        ))
    return findings


def _fmt(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


# --------------------------------------------------------------------------- #
# Commitments: Spanish wording that promises/guarantees/admits something must
# be licensed by matching wording in the English, or it was added in
# translation — the highest-risk error class for a support desk.
# --------------------------------------------------------------------------- #

# (label, Spanish trigger, English licence). Licence = None means "never OK".
_COMMITMENT_RULES: list[tuple[str, str, str | None]] = [
    ("a guarantee",              r"garant\w+",                                  r"guarant|warrant"),
    ("a refund promise",         r"reembols\w+|devoluci[oó]n de(l)? dinero|devol\w+[^.!?]{0,40}dinero",
                                 r"refund|reimburs|money.?back"),
    ("a discount offer",         r"descuent\w+|rebaja\w*",                      r"discount|rebate|reduced price"),
    ("an immediacy promise",     r"de inmediato|inmediatamente|ahora mismo|hoy mismo|cuanto antes",
                                 r"immediat|right now|right away|today|as soon as"),
    ("a definitive-fix claim",   r"definitiv\w+|permanentemente|para siempre",  r"definit|permanent|forever|final"),
    ("a legal admission",        r"responsabilidad legal|legalmente|asumi\w+ (toda )?la responsabilidad|"
                                 r"(la )?culpa (fue )?(totalmente )?nuestra|nuestra culpa|asumi\w+ (todas? )?las consecuencias",
                                 r"legal|liabilit|responsib|our fault|at fault"),
    ("an explicit promise",      r"promet\w+",                                  r"promis"),
    ("a compensation offer",     r"compensa\w+|indemniza\w+",                   r"compensat|indemnif"),
    ("a free-of-charge offer",   r"gratis|gratuit\w+|sin costo|sin cargo|"
                                 r"sin (que tenga que |tener que )?pagar|no (tendr[aá]n? |tienen? )?que pagar",
                                 r"\bfree\b|no charge|no cost|not (have to|need to) pay"),
    ("a cancellation statement", r"cancela\w+|anulad\w+",                       r"cancel|annul"),
    ("a deadline commitment",    r"antes de la?s? \d|a m[aá]s tardar|fecha l[ií]mite",
                                 r"\bby \d|\bby \w+day\b|\bby (january|february|march|april|may|june|july"
                                 r"|august|september|october|november|december)\b|before \d|no later|deadline"),
    # \bplazo: word boundary so 'reemplazo' (replacement) can never trigger it.
    ("a time-window commitment", r"\bplazos?\b",
                                 r"within|\bdays?\b|\bhours?\b|period|deadline|window|closed?\b"),
    # Blame-shifting onto the customer is a business risk on par with a legal
    # admission — and it reverses an apologetic English message's meaning.
    ("an accusation of the customer",
                                 r"se equivoc[oó]|usted cometi[oó]|(la )?culpa (es )?suya|es su culpa|por su error",
                                 r"you made|your (mistake|error|fault)|you were wrong|incorrectly"),
    # 'embarrassed' -> 'embarazada' (pregnant) is the classic EN->ES false
    # cognate: fluent-looking, meaning-destroying, and trivially detectable.
    ("a false-cognate meaning error",
                                 r"embaraza\w+",                                r"pregnan"),
]


def _check_commitments(english: str, spanish: str) -> list[GateFinding]:
    en, es = english.lower(), spanish.lower()
    findings = []
    for label, trigger, licence in _COMMITMENT_RULES:
        match = re.search(trigger, es)
        if not match:
            continue
        if licence and re.search(licence, en):
            continue  # the English says it too — not added by the translation
        findings.append(GateFinding(
            f"{label} ('{match.group()}') appears in the Spanish with no basis in the English",
            SendRecommendation.DO_NOT_SEND,
        ))
    return findings


# --------------------------------------------------------------------------- #
# Negation: English negates but the Spanish doesn't -> the meaning may be
# REVERSED ("has not been charged" -> "ha sido cargada"). Spanish negation
# almost always uses no/nunca/jamás/ni, so its total absence is a hard signal.
# The reverse direction (Spanish adds a negation) is only Review First: polite
# Spanish idioms legitimately add one ("no dude en contactarnos").
# --------------------------------------------------------------------------- #

# Includes common morphological negatives ("unused", "without") — English often
# negates inside a word where Spanish must use "no" ("if unused" -> "si no se usa"),
# and missing those would make faithful translations look like added negations.
_EN_NEG = re.compile(
    r"\b(not|no|never|cannot|without|"
    r"un(?:used|able|available|authorized|authorised|successful|paid|delivered|resolved|confirmed|verified)|"
    r"won't|don't|doesn't|isn't|wasn't|hasn't|haven't|didn't)\b|n't\b",
    re.IGNORECASE,
)
# 'sin' (without) counts as negation — "at no cost" is faithfully rendered as
# "sin costo", and missing it made faithful pairs look like dropped negations.
_ES_NEG = re.compile(r"\b(no|nunca|jam[aá]s|ni|tampoco|nada|ning[uú]n\w*|sin)\b", re.IGNORECASE)


def _check_negation(english: str, spanish: str) -> list[GateFinding]:
    en_neg = bool(_EN_NEG.search(english))
    es_neg = bool(_ES_NEG.search(spanish))
    if en_neg and not es_neg:
        return [GateFinding(
            "the English contains a negation but the Spanish has none — meaning may be reversed",
            SendRecommendation.DO_NOT_SEND,
        )]
    if es_neg and not en_neg:
        return [GateFinding(
            "the Spanish adds a negation that is not in the English",
            SendRecommendation.REVIEW_FIRST,
        )]
    return []


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_safety_gate(english: str, spanish: str) -> list[GateFinding]:
    """Run all deterministic checks on a pair. Returns findings (may be empty)."""
    return (
        _check_numbers(english, spanish)
        + _check_commitments(english, spanish)
        + _check_negation(english, spanish)
    )


def escalate(
    result: TranslationQualityResult,
    findings: list[GateFinding],
    tag: str,
) -> TranslationQualityResult:
    """Merge findings into a result, escalate-only. ``tag`` names the layer
    (e.g. "safety gate", "semantic check") so the operator sees which check
    fired. Never loosens the recommendation. Findings are appended to
    risky_phrases either way, so they are visible even without escalation.
    """
    if not findings:
        return result

    floor = max((f.severity for f in findings), key=_STRICTNESS.get)
    reasons = [f.reason for f in findings]
    update: dict = {
        "risky_phrases": result.risky_phrases + [f"[{tag}] {r}" for r in reasons],
    }

    if _STRICTNESS[floor] > _STRICTNESS[result.send_recommendation]:
        verdict = "Do not send" if floor is SendRecommendation.DO_NOT_SEND else "Review first"
        update["send_recommendation"] = floor
        update["short_summary"] = f"{verdict}: {reasons[0]}."
        update["explanation"] = (
            f"{result.explanation} [{tag.capitalize()}] Escalated from "
            f"'{result.send_recommendation.value}': {'; '.join(reasons)}."
        ).strip()

    return result.model_copy(update=update)


def apply_safety_gate(
    result: TranslationQualityResult, english: str, spanish: str
) -> TranslationQualityResult:
    """Escalate the model's recommendation if the gate found anything worse."""
    return escalate(result, run_safety_gate(english, spanish), "safety gate")
