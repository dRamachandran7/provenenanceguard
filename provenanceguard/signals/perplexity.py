"""Signal 3: Perplexity.

Premise (planning.md): AI text is more *predictable* to a language model than
human text. Perplexity quantifies that predictability:

    PPL(W) = exp( -(1/N) * sum_{i=1..N} ln P(w_i | w_1, ..., w_{i-1}) )

Low perplexity  => the LM was rarely surprised  => leans AI.
High perplexity => the LM was often surprised   => leans human.

Why a local model and not Groq: the formula needs per-token log-probabilities
of the *submitted* text. Groq exposes neither `logprobs` (unsupported on the
chat models) nor an `echo` completions endpoint to score input tokens, so true
perplexity is impossible through it. We instead score tokens with a local
distilGPT-2 via teacher forcing, which lets us compute the formula exactly.

``token_perplexity`` is a pure implementation of the equation and is the part
under test; the distilGPT-2 forward pass merely supplies the log-probabilities.
The scorer is injectable so tests run without the model.

Known limitation: perplexity separates generic AI *prose* from human writing
well, but creative/rhyming AI *poetry* can be high-perplexity (surprising word
choices) and read as human here. That is why this is one of three signals --
the LLM vibe check reliably catches AI poems that perplexity misses.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

from provenanceguard.signals import SignalResult

SIGNAL_NAME = "perplexity"
NEUTRAL_FALLBACK = 0.5
DEFAULT_MODEL = "distilgpt2"

# AI-score mapping thresholds, in *log*-perplexity space (perplexity is roughly
# log-normal, so log space linearises the mapping). Calibrated against labeled
# human/AI samples: human log-PPL ran high, AI ran low.
_LOG_PPL_HUMAN = 4.1  # >= this (PPL ~60) => confidently human (score 0.0)
_LOG_PPL_AI = 3.2     # <= this (PPL ~25) => confidently AI    (score 1.0)

# Reliability ramp by token count: too few tokens => unreliable => held neutral.
_REL_FLOOR_TOKENS = 10
_REL_SPAN_TOKENS = 50

# A scorer maps text -> per-token natural-log probabilities ln P(w_i | w_<i).
LogprobScorer = Callable[[str], List[float]]


def token_perplexity(token_logprobs: List[float]) -> float:
    """Exact implementation of the perplexity formula.

    Args:
        token_logprobs: ln P(w_i | w_<i) for each scored token.

    Returns:
        PPL(W) = exp(-mean(token_logprobs)).
    """
    if not token_logprobs:
        raise ValueError("token_perplexity requires at least one log-probability")
    n = len(token_logprobs)
    mean_ln_p = sum(token_logprobs) / n
    return math.exp(-mean_ln_p)


# --- distilGPT-2 scorer (default provider) -------------------------------

_model = None
_tokenizer = None


def _load_model(model_name: str):
    """Lazily load and cache the model + tokenizer (expensive: once per process)."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.eval()
    return _model, _tokenizer


def gpt2_logprobs(text: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """Per-token ln P(w_i | w_<i) for ``text`` under a local causal LM.

    Uses teacher forcing: a single forward pass yields the distribution at every
    position; we read off the log-prob the model assigned to the token that
    actually followed. Returns [] if the text is shorter than 2 tokens (no
    conditional probability is defined for a lone token).
    """
    import torch

    model, tokenizer = _load_model(model_name)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] < 2:
        return []

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, T, vocab)

    # Align: position i predicts token i+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    chosen = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return chosen[0].tolist()


# --- scoring helpers -----------------------------------------------------


def _linear_map(value: float, human_at: float, ai_at: float) -> float:
    """value >= human_at -> 0.0; value <= ai_at -> 1.0; linear between."""
    if human_at == ai_at:
        return NEUTRAL_FALLBACK
    score = (human_at - value) / (human_at - ai_at)
    return max(0.0, min(1.0, score))


def _score_from_ppl(ppl: float) -> float:
    """Map perplexity to an AI sub-score (low PPL => high AI probability)."""
    return _linear_map(math.log(ppl), _LOG_PPL_HUMAN, _LOG_PPL_AI)


def _reliability(n_tokens: int) -> float:
    return max(0.0, min(1.0, (n_tokens - _REL_FLOOR_TOKENS) / _REL_SPAN_TOKENS))


def analyze_perplexity(
    text: str,
    *,
    scorer: Optional[LogprobScorer] = None,
    model_name: str = DEFAULT_MODEL,
) -> SignalResult:
    """Run the perplexity signal over ``text``.

    Args:
        text: Content to analyze.
        scorer: Optional injected log-prob provider (used in tests). Defaults to
            the local distilGPT-2 scorer.
        model_name: Causal LM to use for the default scorer.

    Returns:
        SignalResult with ``score`` = P(AI-generated). Degrades to a neutral
        ``ok=False`` result if the model is unavailable or the text is too short.
    """
    if not text or not text.strip():
        return SignalResult(
            name=SIGNAL_NAME,
            score=NEUTRAL_FALLBACK,
            reasoning="Empty input; cannot analyze.",
            ok=False,
        )

    scorer = scorer or (lambda t: gpt2_logprobs(t, model_name))
    try:
        token_logprobs = scorer(text)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully (e.g. no model)
        return SignalResult(
            name=SIGNAL_NAME,
            score=NEUTRAL_FALLBACK,
            reasoning=f"Perplexity signal unavailable: {exc}",
            details={"error": str(exc), "model": model_name},
            ok=False,
        )

    n_tokens = len(token_logprobs)
    if n_tokens == 0:
        return SignalResult(
            name=SIGNAL_NAME,
            score=NEUTRAL_FALLBACK,
            reasoning="Too few tokens to compute perplexity.",
            details={"n_tokens": 0, "model": model_name},
            ok=False,
        )

    ppl = token_perplexity(token_logprobs)
    raw_score = _score_from_ppl(ppl)
    reliability = _reliability(n_tokens)
    effective = NEUTRAL_FALLBACK + (raw_score - NEUTRAL_FALLBACK) * reliability

    if reliability < 0.25:
        reasoning = (
            f"Only {n_tokens} tokens; perplexity (PPL={ppl:.1f}) is unreliable on "
            f"short text, so the score is held near neutral."
        )
    else:
        lean = "AI" if effective >= 0.6 else "human" if effective <= 0.4 else "uncertain"
        reasoning = (
            f"Perplexity={ppl:.1f} (lower = more predictable); leans {lean}."
        )

    return SignalResult(
        name=SIGNAL_NAME,
        score=effective,
        reasoning=reasoning,
        details={
            "perplexity": ppl,
            "log_perplexity": math.log(ppl),
            "raw_score": raw_score,
            "reliability": reliability,
            "confidence": reliability,  # consumed by the M4 scorer
            "n_tokens": n_tokens,
            "model": model_name,
        },
    )
