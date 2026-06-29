"""ProvenanceGuard Flask application.

Public API:

    POST /submit       -> run the detection pipeline, return attribution + label
    POST /submit_text  -> alias of /submit (planning.md name)
    POST /appeal       -> file an appeal against a prior decision
    GET  /log          -> view the structured audit log
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, current_app, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenanceguard import audit
from provenanceguard.signals.llm_classifier import classify_with_llm
from provenanceguard.signals.perplexity import analyze_perplexity
from provenanceguard.signals.scorer import combine as combine_signals
from provenanceguard.signals.stylometric import analyze_stylometry
from provenanceguard.transparency import label_result

load_dotenv()

app = Flask(__name__)

# Rate limiting: 10 submissions per minute and 50 per hour per IP address.
# The minute limit absorbs normal interactive bursts; the hourly cap prevents
# sustained automated submissions without blocking any realistic human use.
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)


@app.errorhandler(429)
def ratelimit_handler(e):
    return (
        jsonify({"error": "Rate limit exceeded. Please slow down your submission rate."}),
        429,
    )


def _run_pipeline(text: str) -> Dict[str, Any]:
    """Run all three detection signals and produce a labelled decision."""
    llm = classify_with_llm(text)
    stylo = analyze_stylometry(text)
    ppl = analyze_perplexity(text)
    signals = [llm, stylo, ppl]

    confidence = combine_signals(signals)
    labelling = label_result(confidence)

    return {
        "confidence": confidence,
        "llm_score": llm.score,
        "stylometric_score": stylo.score,
        "perplexity_score": ppl.score,
        "attribution": _attribution(llm.score),
        "label": labelling["label"],
        "label_text": labelling["label_text"],
        "signals": [s.to_dict() for s in signals],
    }


def _attribution(llm_score: float) -> str:
    if llm_score >= 0.6:
        return "likely_ai"
    if llm_score <= 0.4:
        return "likely_human"
    return "uncertain"


@app.post("/submit")
@app.post("/submit_text")
@limiter.limit("10 per minute; 50 per hour", exempt_when=lambda: current_app.testing)
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

    audit.append(
        {
            "event": "submission",
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": audit.now_iso(),
            "text": text,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "llm_score": result["llm_score"],
            "stylometric_score": result["stylometric_score"],
            "perplexity_score": result["perplexity_score"],
            "label": result["label"],
            "label_text": result["label_text"],
            "signals": result["signals"],
            "status": "classified",
        }
    )

    return jsonify(
        {
            "status": "classified",
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "label": result["label"],
            "label_text": result["label_text"],
            "signals": result["signals"],
        }
    )


@app.post("/appeal")
def appeal():
    """File an appeal against a prior decision."""
    payload = request.get_json(silent=True) or {}
    content_id = payload.get("content_id")
    creator_reasoning = payload.get("creator_reasoning", "")

    original = audit.find_submission(content_id) if content_id else None
    if original is None:
        return jsonify({"error": "Unknown content_id."}), 404
    if not creator_reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    previous_label = original.get("label")
    # Content flagged as ai-generated or uncertain is queued for human review.
    queued = previous_label in ("ai-generated", "uncertain")

    audit.append(
        {
            "event": "appeal",
            "content_id": content_id,
            "creator_id": original.get("creator_id"),
            "timestamp": audit.now_iso(),
            "creator_reasoning": creator_reasoning,
            "previous_label": previous_label,
            "queued_for_review": queued,
            "status": "under_review",
        }
    )
    return jsonify(
        {
            "status": "under_review",
            "content_id": content_id,
            "queued_for_review": queued,
        }
    )


@app.get("/log")
def get_log():
    """Return the structured audit log (most recent last)."""
    limit = request.args.get("limit", type=int)
    return jsonify({"entries": audit.read(limit)})


if __name__ == "__main__":
    app.run(debug=True)
