"""Signal 2: Stylometric heuristics (pure Python, no network).

Premise (planning.md): human writing is *bursty and variable*; AI writing is
*smooth and uniform*. We compute measurable statistical properties of the text,
map each to an AI-probability sub-score in [0, 1], and combine them with fixed,
documented weights.

Metrics and why we chose them (weights set by calibration -- see _WEIGHTS):

  * sentence_burstiness  - coefficient of variation (std/mean) of sentence/line
        word counts. Humans mix long and short lines; AI tends to even them out.
        On our calibration set this separated the classes most cleanly
        (human CV ~0.14-0.34 vs AI ~0.0-0.09), so it is the dominant signal.
  * punctuation_variety  - count of distinct punctuation marks used. Humans
        reach for em-dashes, semicolons, parentheticals; basic AI prose sticks
        to commas and periods. A weaker but real separator (human ~2-3 vs
        AI ~1-2). We score variety rather than raw density because density's
        direction is ambiguous. Supporting signal.
  * lexical_diversity    - MATTR (moving-average type-token ratio), a
        length-robust replacement for raw TTR. Premise was "AI is less
        diverse," but it did NOT separate our samples, so it is computed and
        reported but currently weighted 0.
  * word_length_burstiness - coefficient of variation of word lengths. Also
        failed to discriminate on our set; computed/reported, weighted 0.

Short-text handling: heuristics are unreliable on tiny inputs, so a
``reliability`` factor derived from word count shrinks the score toward the
neutral 0.5. A single-line poem therefore contributes almost nothing on its
own and is left to the LLM signals, exactly as planning.md anticipates.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from provenanceguard.signals import SignalResult

SIGNAL_NAME = "stylometric"

NEUTRAL_FALLBACK = 0.5

# MATTR sliding-window size (words).
_MATTR_WINDOW = 25

# Punctuation marks counted toward "variety".
_PUNCT_CHARS = set(";:—–-()\"'!?…,.")

# Word count at/below which reliability is 0, and the span over which it ramps
# to 1.0. Tuned so a ~10-word poem is fully neutral and ~80 words is fully
# trusted.
_REL_FLOOR_WORDS = 10
_REL_SPAN_WORDS = 70

# Metric weights. Renormalized over computable metrics.
#
# Weights were set by calibrating against a set of labeled human/AI samples
# (see report). Only sentence_burstiness and punctuation_variety separated the
# classes. lexical_diversity (MATTR) and word_length_burstiness did NOT
# discriminate on our set -- AI samples were not measurably less diverse -- so
# they are computed and surfaced in `details` for transparency and future
# tuning, but weighted 0 for now. Do not assume they are inert forever: revisit
# with a larger labeled dataset.
_WEIGHTS = {
    "sentence_burstiness": 0.70,
    "punctuation_variety": 0.30,
    "lexical_diversity": 0.0,
    "word_length_burstiness": 0.0,
}

# Threshold pairs for linear mapping: (value_that_means_human, value_that_means_ai).
# Tuned to observed metric ranges: human sentence CV ~0.14-0.34, AI ~0.0-0.09;
# human punctuation variety ~2-3, AI ~1-2.
_THRESHOLDS = {
    # Higher CV => more human.
    "sentence_burstiness": (0.30, 0.05),
    # Higher diversity => more human (unscored; see _WEIGHTS).
    "lexical_diversity": (0.82, 0.60),
    # Higher CV => more human (unscored; see _WEIGHTS).
    "word_length_burstiness": (0.55, 0.30),
    # More distinct marks => more human.
    "punctuation_variety": (3.0, 1.0),
}


def _words(text: str) -> List[str]:
    """Alphabetic word tokens (apostrophes kept for contractions)."""
    return re.findall(r"[A-Za-z']+", text)


def _segments(text: str) -> List[int]:
    """Word counts of each sentence/line.

    Splits on sentence terminators and newlines, since poems often use line
    breaks rather than punctuation as the rhythmic unit.
    """
    parts = re.split(r"[.!?]+|\n+", text)
    return [len(_words(p)) for p in parts if _words(p)]


def _cv(values: List[int]) -> Optional[float]:
    """Population coefficient of variation (std/mean). None if undefined."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (var ** 0.5) / mean


def _mattr(words: List[str]) -> Optional[float]:
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    n = len(words)
    if n == 0:
        return None
    lower = [w.lower() for w in words]
    if n <= _MATTR_WINDOW:
        return len(set(lower)) / n
    ratios = [
        len(set(lower[i : i + _MATTR_WINDOW])) / _MATTR_WINDOW
        for i in range(n - _MATTR_WINDOW + 1)
    ]
    return sum(ratios) / len(ratios)


def _punct_variety(text: str) -> int:
    return len({c for c in text if c in _PUNCT_CHARS})


def _linear_map(value: float, human_at: float, ai_at: float) -> float:
    """Map a metric value to an AI sub-score in [0, 1].

    value >= human_at -> 0.0; value <= ai_at -> 1.0; linear in between.
    Works regardless of whether human_at is above or below ai_at.
    """
    if human_at == ai_at:
        return NEUTRAL_FALLBACK
    score = (human_at - value) / (human_at - ai_at)
    return max(0.0, min(1.0, score))


def _reliability(n_words: int) -> float:
    return max(0.0, min(1.0, (n_words - _REL_FLOOR_WORDS) / _REL_SPAN_WORDS))


def analyze_stylometry(text: str) -> SignalResult:
    """Run the stylometric signal over ``text``.

    Returns a SignalResult whose ``score`` is P(AI-generated). On empty input
    the result is neutral with ``ok=False``. Per-metric values, sub-scores,
    weights, and the reliability factor are exposed in ``details``.
    """
    if not text or not text.strip():
        return SignalResult(
            name=SIGNAL_NAME,
            score=NEUTRAL_FALLBACK,
            reasoning="Empty input; cannot analyze.",
            ok=False,
        )

    words = _words(text)
    n_words = len(words)

    raw_values = {
        "sentence_burstiness": _cv(_segments(text)),
        "lexical_diversity": _mattr(words),
        "word_length_burstiness": _cv([len(w) for w in words]),
        "punctuation_variety": float(_punct_variety(text)),
    }

    # Map each computable metric to a sub-score and accumulate weighted sum.
    subscores = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for name, value in raw_values.items():
        if value is None:
            subscores[name] = None
            continue
        human_at, ai_at = _THRESHOLDS[name]
        sub = _linear_map(value, human_at, ai_at)
        subscores[name] = sub
        weighted_sum += sub * _WEIGHTS[name]
        total_weight += _WEIGHTS[name]

    raw_score = weighted_sum / total_weight if total_weight > 0 else NEUTRAL_FALLBACK

    # Shrink toward neutral for short / unreliable inputs.
    reliability = _reliability(n_words)
    effective = NEUTRAL_FALLBACK + (raw_score - NEUTRAL_FALLBACK) * reliability

    reasoning = _build_reasoning(effective, reliability, raw_values, n_words)

    return SignalResult(
        name=SIGNAL_NAME,
        score=effective,
        reasoning=reasoning,
        details={
            "raw_score": raw_score,
            "reliability": reliability,
            "confidence": reliability,  # consumed by the M4 scorer
            "n_words": n_words,
            "metrics": raw_values,
            "subscores": subscores,
            "weights": _WEIGHTS,
        },
    )


def _build_reasoning(
    effective: float,
    reliability: float,
    raw_values: dict,
    n_words: int,
) -> str:
    if reliability < 0.25:
        return (
            f"Text is short ({n_words} words); stylometry is unreliable, so the "
            f"score is held near neutral and the LLM signals should dominate."
        )
    lean = "AI" if effective >= 0.6 else "human" if effective <= 0.4 else "uncertain"
    cv = raw_values.get("sentence_burstiness")
    cv_txt = f"{cv:.2f}" if cv is not None else "n/a"
    return (
        f"Stylometry leans {lean}: sentence-length burstiness (CV={cv_txt}) is the "
        f"dominant factor; lower variability indicates more uniform, AI-like text."
    )
