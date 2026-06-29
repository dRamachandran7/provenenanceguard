"""Confidence scorer: combines three signal results into one score.

Planning.md spec: "stylometric heuristics and perplexity should be emphasized
over the llm-as-a-judge result, as it is more deterministic."

Base weights — LLM: 0.25, Stylometric: 0.375, Perplexity: 0.375.

Each base weight is scaled by the signal's reported reliability before the
weighted average is taken. This means a short text, where stylometric and
perplexity both pull toward neutral with reliability≈0, lets the LLM signal
carry the full decision rather than being swamped by neutral anchors.
"""

from __future__ import annotations

from typing import List

from provenanceguard.signals import SignalResult

# Weights reflecting planning.md guidance: deterministic signals > LLM judge.
_BASE_WEIGHTS: dict[str, float] = {
    "llm_classifier": 0.25,
    "stylometric": 0.375,
    "perplexity": 0.375,
}


def _reliability(signal: SignalResult) -> float:
    """Return [0, 1] effective reliability for one signal.

    ok=False signals contribute nothing (reliability=0). Signals that expose a
    'reliability' detail field use it; signals with no such field (LLM) default
    to 1.0 when ok=True — they always return a definitive judgment.
    """
    if not signal.ok:
        return 0.0
    return float(signal.details.get("reliability", 1.0))


def combine(signals: List[SignalResult]) -> float:
    """Compute a reliability-weighted confidence score from signal results.

    Returns a float in [0, 1] representing P(AI-generated). Falls back to a
    simple mean if every signal is unusable (ok=False).
    """
    weighted_sum = 0.0
    weight_total = 0.0

    for s in signals:
        base = _BASE_WEIGHTS.get(s.name, 0.0)
        effective_weight = base * _reliability(s)
        weighted_sum += s.score * effective_weight
        weight_total += effective_weight

    if weight_total == 0.0:
        return sum(s.score for s in signals) / len(signals) if signals else 0.5

    return weighted_sum / weight_total
