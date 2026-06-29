"""Tests for Signal 3: perplexity.

The formula and mapping are tested with injected log-probabilities (fast, no
model). One guarded integration test exercises the real distilGPT-2 scorer.
"""

import math

import pytest

from provenanceguard.signals.perplexity import (
    NEUTRAL_FALLBACK,
    SIGNAL_NAME,
    _score_from_ppl,
    analyze_perplexity,
    token_perplexity,
)


# --- The formula itself: PPL = exp(-mean(ln P)) --------------------------


def test_perplexity_of_uniform_probability_is_inverse_probability():
    # If every token has probability p, PPL must equal 1/p.
    for p in (0.5, 0.25, 0.1):
        logprobs = [math.log(p)] * 7
        assert token_perplexity(logprobs) == pytest.approx(1 / p)


def test_perfect_prediction_has_perplexity_one():
    # ln P = 0 means P = 1 for every token.
    assert token_perplexity([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_perplexity_matches_hand_computed_mixed_case():
    logprobs = [math.log(0.5), math.log(0.25)]  # mean ln = (ln.5+ln.25)/2
    expected = math.exp(-(math.log(0.5) + math.log(0.25)) / 2)
    assert token_perplexity(logprobs) == pytest.approx(expected)


def test_more_surprise_means_higher_perplexity():
    confident = token_perplexity([math.log(0.9)] * 5)
    surprised = token_perplexity([math.log(0.1)] * 5)
    assert surprised > confident


def test_empty_logprobs_raises():
    with pytest.raises(ValueError):
        token_perplexity([])


# --- PPL -> AI score mapping ---------------------------------------------


def test_low_perplexity_maps_to_ai():
    assert _score_from_ppl(20.0) == pytest.approx(1.0)  # log(20)=3.0 <= 3.2


def test_high_perplexity_maps_to_human():
    assert _score_from_ppl(150.0) == 0.0  # log(150)=5.0 >= 4.1


def test_mapping_is_monotonic_and_clamped():
    assert _score_from_ppl(10.0) >= _score_from_ppl(60.0) >= _score_from_ppl(300.0)
    assert 0.0 <= _score_from_ppl(45.0) <= 1.0


# --- analyze_perplexity with an injected scorer --------------------------


def _scorer(logprobs):
    return lambda text: list(logprobs)


def test_predictable_text_scores_ai():
    # Low perplexity + enough tokens for full reliability.
    r = analyze_perplexity("x", scorer=_scorer([math.log(0.8)] * 60))
    assert r.ok is True
    assert r.score > 0.6
    assert r.details["perplexity"] == pytest.approx(1 / 0.8)


def test_surprising_text_scores_human():
    r = analyze_perplexity("x", scorer=_scorer([math.log(0.01)] * 60))
    assert r.score < 0.4


def test_short_text_is_pulled_to_neutral():
    r = analyze_perplexity("x", scorer=_scorer([math.log(0.8)] * 5))
    assert r.score == pytest.approx(NEUTRAL_FALLBACK)
    assert r.details["reliability"] == 0.0


def test_empty_text_is_neutral_and_not_ok():
    r = analyze_perplexity("   ", scorer=_scorer([math.log(0.5)] * 30))
    assert r.ok is False
    assert r.score == NEUTRAL_FALLBACK


def test_too_few_tokens_returns_not_ok():
    r = analyze_perplexity("hi", scorer=_scorer([]))
    assert r.ok is False
    assert r.details["n_tokens"] == 0


def test_scorer_failure_degrades_gracefully():
    def boom(text):
        raise RuntimeError("model not loaded")

    r = analyze_perplexity("text", scorer=boom)
    assert r.ok is False
    assert r.score == NEUTRAL_FALLBACK
    assert "model not loaded" in r.details["error"]


def test_result_shape():
    r = analyze_perplexity("x", scorer=_scorer([math.log(0.5)] * 40))
    assert r.name == SIGNAL_NAME
    assert {"perplexity", "log_perplexity", "confidence", "n_tokens"} <= set(r.details)


# --- Real distilGPT-2 integration (guarded) ------------------------------

_AI_GENERIC = (
    "Reading is a wonderful habit that brings many benefits to our lives. "
    "It expands our knowledge and helps us understand the world around us. "
    "Books allow us to explore new ideas and grow as people every day."
)
_HUMAN_FREE = (
    "I celebrate myself, and sing myself, and what I assume you shall assume, "
    "for every atom belonging to me as good belongs to you."
)


def test_real_distilgpt2_ranks_generic_ai_above_human_verse():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    ai = analyze_perplexity(_AI_GENERIC)
    human = analyze_perplexity(_HUMAN_FREE)
    if not (ai.ok and human.ok):
        pytest.skip("distilGPT-2 unavailable in this environment")
    assert ai.score > human.score
