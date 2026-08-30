"""
feedback.py — Stage 7 of the Lens pipeline: Feedback

Non-LLM. Implements the "lightweight closed loop" design from the original
handoff (section 6): NOT full ML retraining, NOT capture-only.

  1. An analyst confirms/rejects/edits a proposed cause -> stored in an
     append-only feedback store.
  2. get_similar_confirmed_cases() pulls the 1-2 most similar past
     CONFIRMED cases for a future Explain call on the same (kpi, cause_type)
     pair, to be injected as few-shot examples.
  3. get_reliability_score() computes a running confirm/reject ratio per
     (kpi, cause_type) -- a cause type that's been wrong before should
     nudge the confidence gate down until it re-earns trust.

Why "cause_type" is Decompose's top_driver, not free text
------------------------------------------------------------
Same reasoning as recommend.py's lever lookup: top_driver (volume_effect /
mix_effect / price_effect / category_shift) is a small, closed, structured
category -- perfect for grouping feedback and computing a reliability
score. Grouping on Explain's free-text proposed cause instead would need
fuzzy matching or another LLM call just to cluster similar causes, which
defeats the point of a lightweight, deterministic feedback loop.

Why this is NOT full ML retraining
--------------------------------------
No fine-tuning, no embeddings, no vector search. Similarity matching is
exact categorical matching on (kpi, cause_type) -- deliberately simple, per
the handoff's own reasoning: there's no historical "ground truth" dataset
of confirmed causes to train a classifier on, so a trained model would
need self-invented labels, less trustworthy than this transparent,
auditable approach.

Run standalone for a quick smoke test:
    python3 src/feedback.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "feedback_store.jsonl")

VALID_VERDICTS = {"confirmed", "rejected", "edited"}


def record_feedback(kpi: str, region: str, segment: str, week: str, cause_type: str,
                     cause_text: str, verdict: str, note: str = None) -> dict:
    """Appends one feedback entry. Append-only -- never truncates or edits
    prior entries, same pattern as governance.py's audit log."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got '{verdict}'")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kpi": kpi, "region": region, "segment": segment, "week": week,
        "cause_type": cause_type, "cause_text": cause_text,
        "verdict": verdict, "note": note,
    }

    Path(FEEDBACK_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_STORE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def _load_all_feedback() -> list:
    if not os.path.exists(FEEDBACK_STORE_PATH):
        return []
    entries = []
    with open(FEEDBACK_STORE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_similar_confirmed_cases(kpi: str, cause_type: str, limit: int = 2) -> list:
    """Returns up to `limit` most recent CONFIRMED cases for this
    (kpi, cause_type) pair, to be injected as few-shot examples into a
    future Explain propose call. Returns [] if there's no history yet --
    callers should treat this as "no few-shot examples available", not
    an error."""
    entries = _load_all_feedback()
    matches = [
        e for e in entries
        if e["kpi"] == kpi and e["cause_type"] == cause_type and e["verdict"] == "confirmed"
    ]
    matches.sort(key=lambda e: e["timestamp"], reverse=True)
    return matches[:limit]


def get_reliability_score(kpi: str, cause_type: str) -> dict:
    """Running confirm/reject ratio for this (kpi, cause_type) pair.
    "edited" entries are stored but excluded from the ratio -- an edit
    means the analyst partially agreed and partially disagreed, which is
    genuinely ambiguous and shouldn't be silently counted either way.

    Returns a neutral 0.5 score with n_confirmed=n_rejected=0 when there's
    no track record yet -- an UNPROVEN cause type should not be penalized
    the same way a cause type with an actual bad track record should be.
    """
    entries = _load_all_feedback()
    matches = [e for e in entries if e["kpi"] == kpi and e["cause_type"] == cause_type]

    n_confirmed = sum(1 for e in matches if e["verdict"] == "confirmed")
    n_rejected = sum(1 for e in matches if e["verdict"] == "rejected")
    n_edited = sum(1 for e in matches if e["verdict"] == "edited")

    total_decisive = n_confirmed + n_rejected
    if total_decisive == 0:
        score = 0.5  # neutral -- no track record yet, not penalized
    else:
        score = n_confirmed / total_decisive

    return {
        "kpi": kpi, "cause_type": cause_type,
        "n_confirmed": n_confirmed, "n_rejected": n_rejected, "n_edited": n_edited,
        "reliability_score": round(score, 3),
        "has_track_record": total_decisive > 0,
    }


def _smoke_test():
    print("feedback.py smoke test (writes to a temp store, does not touch the real one)\n")

    import tempfile
    global FEEDBACK_STORE_PATH
    original_path = FEEDBACK_STORE_PATH
    FEEDBACK_STORE_PATH = os.path.join(tempfile.gettempdir(), "lens_feedback_smoketest.jsonl")
    if os.path.exists(FEEDBACK_STORE_PATH):
        os.remove(FEEDBACK_STORE_PATH)

    print("Recording 3 feedback entries for revenue/volume_effect (2 confirmed, 1 rejected)...")
    record_feedback("revenue", "East", "Corporate", "2017-12-11", "volume_effect",
                     "Volume drop drove the revenue decline", "confirmed")
    record_feedback("revenue", "West", "Consumer", "2016-12-26", "volume_effect",
                     "Volume drop drove the revenue decline", "confirmed")
    record_feedback("revenue", "Central", "Consumer", "2017-12-25", "volume_effect",
                     "Volume drop drove the revenue decline", "rejected")

    print("\nSimilar confirmed cases for (revenue, volume_effect):")
    similar = get_similar_confirmed_cases("revenue", "volume_effect", limit=2)
    for s in similar:
        print(f"  {s['region']}/{s['segment']} {s['week']}: \"{s['cause_text']}\"")

    print("\nReliability score for (revenue, volume_effect):")
    reliability = get_reliability_score("revenue", "volume_effect")
    print(f"  {reliability}")
    assert reliability["reliability_score"] == round(2/3, 3), "Expected 2/3 confirm rate"
    print("  PASS: 2 confirmed / 1 rejected = 0.667 reliability, as expected.")

    print("\nReliability score for an UNSEEN (kpi, cause_type) pair (should be neutral 0.5):")
    unseen = get_reliability_score("profit_margin", "price_effect")
    print(f"  {unseen}")
    assert unseen["reliability_score"] == 0.5 and not unseen["has_track_record"]
    print("  PASS: unproven cause type correctly defaults to neutral, not penalized.")

    os.remove(FEEDBACK_STORE_PATH)
    FEEDBACK_STORE_PATH = original_path
    print("\nSmoke test complete, temp file cleaned up.")


if __name__ == "__main__":
    _smoke_test()
