"""Detection signals for the attribution pipeline.

Each signal takes a piece of text and returns a SignalResult whose ``score``
is an estimated probability in [0.0, 1.0] that the text is AI-generated
(0.0 = confidently human, 1.0 = confidently AI).
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SignalResult:
    """The output of a single detection signal.

    Attributes:
        name: Identifier of the signal that produced this result.
        score: Estimated P(AI-generated) in [0.0, 1.0].
        reasoning: Human-readable explanation of the score.
        details: Signal-specific extra data (raw response, sub-metrics, etc.).
        ok: False when the signal failed and ``score`` is a neutral fallback.
    """

    name: str
    score: float
    reasoning: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True

    def __post_init__(self) -> None:
        # Defensively clamp so a misbehaving signal can never poison the scorer.
        self.score = max(0.0, min(1.0, float(self.score)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "reasoning": self.reasoning,
            "details": self.details,
            "ok": self.ok,
        }
