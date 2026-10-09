"""Pydantic schemas — the contract between the AI model and our code.

Every model response is validated against these schemas BEFORE we display or
store anything. If the model returns something off-spec (wrong field, bad type,
an issue_type that isn't allowed, a score over 100), validation fails and the
caller re-asks the model. This is what stops bad output from reaching a real
service-request system.

Project 2A uses ``ServiceRequestResult``.
Project 3A uses ``TranslationQualityResult``.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Shared normaliser helpers (real models return slightly-off shapes).
# --------------------------------------------------------------------------- #

def _clamp_score(v) -> int:
    """Accept '85', 85.0, etc.; clamp into 0-100. Junk becomes 0."""
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _as_list(v):
    """A model that returns a bare string where a list is expected gets wrapped."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    return v


def _match_enum(v, enum_cls, default: str):
    """Accept any casing / underscores / hyphens; map unknown labels to a safe
    default instead of erroring (e.g. "do_not_send" -> "Do Not Send")."""
    if isinstance(v, str):
        cleaned = re.sub(r"[_\-]+", " ", v)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().title()
        valid = {e.value for e in enum_cls}
        return cleaned if cleaned in valid else default
    return v


class IssueType(str, Enum):
    """The fixed set of issue categories from the spec. Anything the model
    returns that isn't on this list is mapped to OTHER (see the validator)."""

    LOGIN = "Login"
    PAYMENT = "Payment"
    REFUND = "Refund"
    ORDER = "Order"
    LOYALTY = "Loyalty"
    TECHNICAL = "Technical"
    DATA = "Data"
    INTEGRATION = "Integration"
    OTHER = "Other"


class ServiceRequestResult(BaseModel):
    """Cleaned, structured version of a messy service request note (Project 2A)."""

    cleaned_description: str
    short_summary: str
    issue_type: IssueType
    customer_impact: str
    technical_details: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    suggested_next_step: str
    tone_adjustment: str
    ml_tags: List[str] = Field(default_factory=list)
    confidence_score: int = Field(ge=0, le=100)

    # --- forgiving normalisers: real models return slightly-off shapes ------

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalise_issue_type(cls, v):
        """Accept any casing; map unknown labels to 'Other' instead of erroring."""
        if isinstance(v, str):
            cleaned = v.strip().title()
            valid = {e.value for e in IssueType}
            return cleaned if cleaned in valid else IssueType.OTHER.value
        return v

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _coerce_score(cls, v):
        """Accept '85', 85.0, etc.; clamp into 0-100."""
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, n))

    @field_validator(
        "technical_details", "missing_information", "ml_tags", mode="before"
    )
    @classmethod
    def _ensure_list(cls, v):
        return _as_list(v)


# --------------------------------------------------------------------------- #
# Project 3A — Translation Quality Checker
# --------------------------------------------------------------------------- #

class AccuracyRating(str, Enum):
    """How faithfully the Spanish conveys the English meaning."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ToneCheck(str, Enum):
    """Customer-service tone of the translation."""

    FRIENDLY = "Friendly"
    PROFESSIONAL = "Professional"
    TOO_HARSH = "Too Harsh"
    TOO_CASUAL = "Too Casual"
    UNCLEAR = "Unclear"


class SendRecommendation(str, Enum):
    """Whether the translation is safe to send to the customer."""

    SEND = "Send"
    REVIEW_FIRST = "Review First"
    DO_NOT_SEND = "Do Not Send"


class TranslationQualityResult(BaseModel):
    """Quality assessment of an English -> Spanish customer-service message (3A)."""

    short_summary: str
    accuracy_rating: AccuracyRating
    confidence_score: int = Field(ge=0, le=100)
    tone_check: ToneCheck
    risky_phrases: List[str] = Field(default_factory=list)
    missing_meaning: List[str] = Field(default_factory=list)
    added_meaning: List[str] = Field(default_factory=list)
    suggested_correction: str = ""
    back_translation: str
    send_recommendation: SendRecommendation
    explanation: str

    # --- forgiving normalisers --------------------------------------------- #

    @field_validator("accuracy_rating", mode="before")
    @classmethod
    def _norm_accuracy(cls, v):
        # If the rating is unparseable, assume the worst (Low) — conservative.
        return _match_enum(v, AccuracyRating, AccuracyRating.LOW.value)

    @field_validator("tone_check", mode="before")
    @classmethod
    def _norm_tone(cls, v):
        return _match_enum(v, ToneCheck, ToneCheck.UNCLEAR.value)

    @field_validator("send_recommendation", mode="before")
    @classmethod
    def _norm_send(cls, v):
        # Never auto-"Send" on an unparseable value — fall back to Review First.
        return _match_enum(v, SendRecommendation, SendRecommendation.REVIEW_FIRST.value)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _coerce_score(cls, v):
        return _clamp_score(v)

    @field_validator(
        "risky_phrases", "missing_meaning", "added_meaning", mode="before"
    )
    @classmethod
    def _ensure_list(cls, v):
        return _as_list(v)

    @field_validator(
        "short_summary", "suggested_correction", "back_translation", "explanation",
        mode="before",
    )
    @classmethod
    def _coerce_text(cls, v):
        """Tolerate a model that sends null for an optional text field."""
        return "" if v is None else v
