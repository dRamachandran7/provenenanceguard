"""Transparency label engine.

Maps a combined confidence score to a machine-readable label and a
plain-language display string shown to platform readers.

Thresholds from planning.md:
    0.0–0.4  → human
    0.4–0.6  → uncertain
    0.6–1.0  → ai-generated

Within each tier, label_text varies with the degree of confidence so that a
score of 0.62 reads meaningfully different from one of 0.93.
"""

from __future__ import annotations

# Each entry: (minimum confidence for this tier, label, display text).
# Evaluated top-to-bottom; first matching floor wins.
_TIERS = [
    (
        0.85,
        "ai-generated",
        "Our analysis strongly suggests this content was written by an AI.",
    ),
    (
        0.60,
        "ai-generated",
        "Our analysis suggests this content was likely written by an AI.",
    ),
    (
        0.40,
        "uncertain",
        "We couldn't confidently determine whether this content was written "
        "by a human or an AI.",
    ),
    (
        0.20,
        "human",
        "Our analysis suggests this content was likely written by a human.",
    ),
    (
        0.00,
        "human",
        "Our analysis strongly suggests this content was written by a human.",
    ),
]


def label_result(confidence: float) -> dict:
    """Return {'label': ..., 'label_text': ...} for the given confidence score."""
    for floor, label, text in _TIERS:
        if confidence >= floor:
            return {"label": label, "label_text": text}
    # unreachable for valid [0,1] input, but safe fallback
    return {"label": "human", "label_text": _TIERS[-1][2]}
