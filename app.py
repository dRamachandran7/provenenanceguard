"""ProvenanceGuard Flask application (skeleton).

Public API:

    POST /submit       -> run the detection pipeline, return attribution + label
    POST /submit_text  -> alias of /submit (planning.md name)
    POST /appeal       -> file an appeal against a prior decision
    GET  /log          -> view the structured audit log

Signals 1 (LLM vibe check) and 2 (stylometric) are implemented. The real
confidence scorer, Signal 3 (perplexity), the transparency-label engine, and
the full appeals review queue are stubbed and marked TODO for upcoming
milestones.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from provenanceguard import audit
from provenanceguard.signals.llm_classifier import classify_with_llm
from provenanceguard.signals.stylometric import analyze_stylometry

load_dotenv()

app = Flask(__name__)


def _run_pipeline(text: str) -> Dict[str, Any]:
    """Run the available detection signals over ``text``.

    TODO(M4): add the perplexity signal and combine via the real confidence
    scorer. For now Signals 1 (LLM) and 2 (stylometric) run and are averaged.
    """
    llm = classify_with_llm(text)
    stylo = analyze_stylometry(text)
    signals = [llm, stylo]

    # TODO(M4): real confidence scorer. Placeholder = mean of signal scores.
    confidence = sum(s.score for s in signals) / len(signals)

    return {
        "confidence": confidence,
        "llm_score": llm.score,
        "stylometric_score": stylo.score,
        # attribution reflects Signal 1 specifically, per the spec.
        "attribution": _attribution(llm.score),
        # TODO(M5): real transparency-label engine per planning.md thresholds.
        "label": _placeholder_label(confidence),
        "signals": [s.to_dict() for s in signals],
    }


def _attribution(llm_score: float) -> str:
    """Map Signal 1's score to a coarse attribution result."""
    if llm_score >= 0.6:
        return "likely_ai"
    if llm_score <= 0.4:
        return "likely_human"
    return "uncertain"


def _placeholder_label(confidence: float) -> str:
    """Temporary label mapping; replaced by the label engine in M5."""
    if confidence < 0.4:
        return "human"
    if confidence < 0.6:
        return "uncertain"
    return "ai-generated"


@app.post("/submit")
@app.post("/submit_text")
def submit():
    """Accept content for attribution analysis and return the decision."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    creator_id = payload.get("creator_id", "")

    if not text or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400
    if not creator_id or not str(creator_id).strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    result = _run_pipeline(text)
    content_id = str(uuid.uuid4())

    entry = {
        "event": "submission",
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": audit.now_iso(),
        "text": text,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "llm_score": result["llm_score"],
        "stylometric_score": result["stylometric_score"],
        "label": result["label"],
        "signals": result["signals"],
        "status": "classified",
    }
    audit.append(entry)

    return jsonify(
        {
            "status": "classified",
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "label": result["label"],
            "signals": result["signals"],
        }
    )


@app.post("/appeal")
def appeal():
    """File an appeal against a prior decision. TODO(M5): full review queue."""
    payload = request.get_json(silent=True) or {}
    content_id = payload.get("content_id")
    reason = payload.get("reason", "")

    original = audit.find_submission(content_id) if content_id else None
    if original is None:
        return jsonify({"error": "Unknown content_id."}), 404
    if not reason.strip():
        return jsonify({"error": "Field 'reason' is required."}), 400

    # TODO(M5): push to human review queue when label is ai-generated/uncertain.
    audit.append(
        {
            "event": "appeal",
            "content_id": content_id,
            "creator_id": original.get("creator_id"),
            "timestamp": audit.now_iso(),
            "reason": reason,
            "previous_label": original.get("label"),
            "status": "under_review",
        }
    )
    return jsonify({"status": "under_review", "content_id": content_id})


@app.get("/log")
def get_log():
    """Return the structured audit log (most recent last)."""
    limit = request.args.get("limit", type=int)
    return jsonify({"entries": audit.read(limit)})


if __name__ == "__main__":
    app.run(debug=True)
