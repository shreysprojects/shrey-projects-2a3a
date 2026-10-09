"""Tests for the 3A deterministic safety gate.

All offline — the gate is pure stdlib regex/string logic, so these tests prove
its exact behaviour with no model or network involved. Each dangerous class from
samples/evaluation/translation_eval_set.json has a corresponding check here, but
phrased differently where possible (the gate must catch the CLASS, not the
memorised sentence).
"""
import pytest

from src.safety_gate import GateFinding, apply_safety_gate, run_safety_gate
from src.schemas import SendRecommendation, TranslationQualityResult


def _result(recommendation="Send", **overrides):
    base = {
        "short_summary": "Safe to send: accurate and professional.",
        "accuracy_rating": "High",
        "confidence_score": 90,
        "tone_check": "Professional",
        "risky_phrases": [],
        "missing_meaning": [],
        "added_meaning": [],
        "suggested_correction": "",
        "back_translation": "(back translation)",
        "send_recommendation": recommendation,
        "explanation": "Looks fine.",
    }
    base.update(overrides)
    return TranslationQualityResult.model_validate(base)


def _severities(findings):
    return {f.severity for f in findings}


# --------------------------------------------------------------------------- #
# Clean pairs: the gate must stay silent (no false alarms on good translations)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("en, es", [
    # faithful, no numbers/commitments/negations
    ("Your order has shipped.", "Su pedido ha sido enviado."),
    # numbers present but MATCHING on both sides
    ("Your refund of $45 will arrive within 5 business days.",
     "Su reembolso de $45 llegará en un plazo de 5 días hábiles."),
    # commitment vocabulary licensed by the English
    ("We guarantee a full refund immediately.",
     "Le garantizamos un reembolso completo de inmediato."),
    # negation on both sides
    ("Your account has not been charged.", "Su cuenta no ha sido cargada."),
    # times/names preserved (eval #21 shape)
    ("Hi Maria, your appointment is confirmed for Monday at 3 PM.",
     "Hola María, su cita está confirmada para el lunes a las 3 de la tarde."),
])
def test_gate_silent_on_clean_pairs(en, es):
    assert run_safety_gate(en, es) == []


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

def test_changed_amount_is_do_not_send():
    # wrong_number class: $45 became $54 (both directions flagged)
    findings = run_safety_gate(
        "Your refund of $45 has been processed.",
        "Su reembolso de $54 ha sido procesado.",
    )
    assert SendRecommendation.DO_NOT_SEND in _severities(findings)


def test_added_number_is_do_not_send():
    # added_deadline class, new phrasing: adds "2 horas" out of nowhere
    findings = run_safety_gate(
        "We will get back to you.",
        "Le responderemos en 2 horas.",
    )
    assert SendRecommendation.DO_NOT_SEND in _severities(findings)


def test_dropped_number_is_review_first():
    findings = run_safety_gate(
        "You can return it within 30 days.",
        "Puede devolverlo pronto.",
    )
    assert _severities(findings) == {SendRecommendation.REVIEW_FIRST}


def test_separator_variants_do_not_false_alarm():
    # 1,000.50 (EN) == 1.000,50 (ES) — same value, different separators
    assert run_safety_gate(
        "A charge of $1,000.50 was reversed.",
        "Se revirtió un cargo de $1.000,50.",
    ) == []


def test_changed_currency_is_do_not_send():
    # Same digits, different currency: £ -> euros
    findings = run_safety_gate(
        "We have refunded £75 to your card.",
        "Hemos reembolsado 75 € a su tarjeta.",
    )
    assert SendRecommendation.DO_NOT_SEND in _severities(findings)


def test_same_currency_written_out_is_allowed():
    # '$' on one side, 'dólares' on the other — same currency, no alarm
    assert run_safety_gate(
        "Your $20 credit has been applied.",
        "Su crédito de 20 dólares ha sido aplicado.",
    ) == []


# --------------------------------------------------------------------------- #
# Commitments
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("en, es", [
    # added_refund_promise class (new phrasing)
    ("We will look into your billing question.",
     "Le haremos un reembolso completo."),
    # overstated_certainty class (new phrasing)
    ("This should fix the problem.", "Esto solucionará el problema de forma garantizada."),
    # added_legal_claim class
    ("We will do our best to resolve this.",
     "Aceptamos toda la responsabilidad legal por este problema."),
    # mistranslated_key_term class: refund -> discount
    ("We can offer you a refund.", "Podemos ofrecerle un descuento."),
    # reversed_meaning class: cancellation invented
    ("Your order has shipped.", "Su pedido fue cancelado."),
    # free-of-charge invented
    ("We will replace the part.", "Reemplazaremos la pieza gratis."),
    # harsh_blaming_tone class: apology became an accusation
    ("We are sorry for the confusion. Could you please confirm your email?",
     "Usted se equivocó. Confirme su correo."),
    # money-back promise phrased without 'reembolso'
    ("We are checking on your order.", "Le devolveremos su dinero mañana."),
    # fault admission phrased without 'legal'
    ("We apologize for the trouble.", "El error fue completamente nuestra culpa."),
    # pay-nothing phrasing of a free offer
    ("A technician will visit you.", "Un técnico le visitará y no tendrá que pagar."),
    # embarrassed -> pregnant false cognate
    ("We are embarrassed about the mix-up.", "Estamos muy embarazados por la confusión."),
])
def test_unlicensed_commitment_is_do_not_send(en, es):
    findings = run_safety_gate(en, es)
    assert SendRecommendation.DO_NOT_SEND in _severities(findings)


def test_licensed_commitment_is_allowed():
    # "plazo de 30 días" is fine when the English says "within 30 days"
    assert run_safety_gate(
        "If unused, you can return it within 30 days for a full refund.",
        "Si no se usa, puede devolverlo en un plazo de 30 días para un reembolso completo.",
    ) == []


@pytest.mark.parametrize("en, es", [
    # 'a más tardar' licensed by "by <weekday>" / "by <month day>" (152-set bug)
    ("We guarantee delivery by Friday, July 17.",
     "Le garantizamos la entrega a más tardar el viernes 17 de julio."),
    ("Please verify your email address by August 1.",
     "Verifique su correo electrónico a más tardar el 1 de agosto."),
    # 'reemplazo' (replacement) must never trigger the 'plazo' rule (152-set bug)
    ("We are sending you a replacement item.",
     "Le enviaremos un reemplazo del artículo."),
    # 'plazo' licensed by "window ... closed" phrasing (152-set bug)
    ("We can't reverse this charge because the dispute window closed on June 3.",
     "No podemos revertir este cargo porque el plazo para disputarlo venció el 3 de junio."),
    # 'sin costo' is a faithful negation for "at no cost" (152-set bug)
    ("If the item arrives damaged, we'll arrange a pickup at no cost.",
     "Si el artículo llega dañado, coordinaremos la recogida sin costo alguno."),
])
def test_152_set_false_alarm_fixes(en, es):
    assert run_safety_gate(en, es) == []


# --------------------------------------------------------------------------- #
# Negation
# --------------------------------------------------------------------------- #

def test_dropped_negation_is_do_not_send():
    # negation_dropped class (new phrasing): "will not be charged" -> "will be charged"
    findings = run_safety_gate(
        "You will not be charged for shipping.",
        "Se le cobrará el envío.",
    )
    assert SendRecommendation.DO_NOT_SEND in _severities(findings)


def test_added_polite_negation_is_only_review_first():
    # Polite Spanish idiom adds "no dude" — must not hard-block
    findings = run_safety_gate(
        "Feel free to contact us.",
        "No dude en contactarnos.",
    )
    assert _severities(findings) == {SendRecommendation.REVIEW_FIRST}


# --------------------------------------------------------------------------- #
# apply_safety_gate: escalate-only merging into the model's result
# --------------------------------------------------------------------------- #

def test_escalates_send_to_do_not_send_and_rewrites_summary():
    result = _result("Send")
    gated = apply_safety_gate(
        result,
        "We will look into your billing question.",
        "Le garantizamos un reembolso completo de inmediato.",
    )
    assert gated.send_recommendation is SendRecommendation.DO_NOT_SEND
    assert gated.short_summary.startswith("Do not send")
    assert any("[safety gate]" in p for p in gated.risky_phrases)
    assert "[Safety gate]" in gated.explanation


def test_never_downgrades_a_stricter_model_verdict():
    result = _result("Do Not Send", short_summary="Do not send: rude tone.")
    gated = apply_safety_gate(result, "Your order has shipped.", "Su pedido ha sido enviado.")
    assert gated.send_recommendation is SendRecommendation.DO_NOT_SEND
    assert gated.short_summary == "Do not send: rude tone."  # untouched


def test_clean_pair_returns_result_unchanged():
    result = _result("Send")
    gated = apply_safety_gate(result, "Your order has shipped.", "Su pedido ha sido enviado.")
    assert gated == result


def test_findings_recorded_even_without_escalation():
    # Model already said Do Not Send; gate agrees — reasons still logged.
    result = _result("Do Not Send", short_summary="Do not send: wrong amount.")
    gated = apply_safety_gate(
        result,
        "Your refund of $45 has been processed.",
        "Su reembolso de $54 ha sido procesado.",
    )
    assert gated.send_recommendation is SendRecommendation.DO_NOT_SEND
    assert any("[safety gate]" in p for p in gated.risky_phrases)
