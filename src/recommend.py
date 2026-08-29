"""
recommend.py — Stage 5 of the Lens pipeline: Recommend

Hybrid stage. Given Explain's output for a scenario, produces a structured
recommendation: driver -> lever -> action -> expected_impact -> owner ->
confidence -> monitoring_plan.

LLM vs non-LLM boundary (explicit, per the brief's requirement)
--------------------------------------------------------------------
  Non-LLM (deterministic):
    - driver -> lever -> owner mapping (LEVER_LOOKUP below)
    - confidence (forced from explain.py's forced_confidence_tier -- never
      re-derived or left to the LLM, same hybrid rule used throughout)
    - monitoring_plan (a fixed template referencing Detect's own re-run cadence)
    - the ENTIRE abstention path (low_confidence_abstain / insufficient_history)
      -- no LLM call is made for these; see rationale below.

  LLM:
    - action text: the specific, concrete recommendation phrasing
    - expected_impact text: phrased using ONLY numbers already present in
      the evidence packet (total_change from decompose.py) -- the LLM is
      not permitted to invent a new estimate, same grounding rule as
      explain.py's propose/challenge steps.

Why the lever lookup keys off decompose.py's top_driver, not Explain's
free-text proposed cause
------------------------------------------------------------------------
top_driver (volume_effect / mix_effect / price_effect / category_shift) is
a small, closed, structured category -- exactly what a deterministic
lookup table needs. Explain's proposed cause is prose; mapping arbitrary
prose to a fixed lever reliably would require ANOTHER LLM call just to
categorize it, which defeats the point of having a deterministic step here
at all. The LLM's job stays narrow: phrase the action, don't decide the
lever.

Why abstention scenarios get zero LLM calls
------------------------------------------------
Same reasoning as explain.py's sparse-history path: recommending a
specific action from insufficient evidence would misrepresent confidence
regardless of how the recommendation is phrased. The honest recommendation
in a low-confidence/insufficient-history case is "gather more evidence
first" -- a deterministic template, not something requiring LLM phrasing.

Run:
    export GROQ_API_KEY=gsk_...  (or set in .env)
    python3 src/recommend.py
"""

import os
import json
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Reuse explain.py's Groq call machinery (telemetry wrapper, JSON parsing,
# pricing table) rather than duplicating it.
import sys
sys.path.insert(0, os.path.dirname(__file__))
from explain import call_groq, _parse_json_response, GROQ_MODEL


# --- Deterministic driver -> lever -> owner lookup (non-LLM) ---
LEVER_LOOKUP = {
    "volume_effect": {"lever": "Demand / retention", "owner": "Sales / Account Management"},
    "price_effect": {"lever": "Pricing", "owner": "Pricing / Revenue team"},
    "mix_effect": {"lever": "Assortment / merchandising", "owner": "Category Management"},
    "category_shift": {"lever": "Category-specific review", "owner": "Category Management"},
}

RECOMMEND_SYSTEM_PROMPT = """You are Lens, writing a specific business action recommendation.

You will be given: the evidence packet, which driver was identified as dominant,
which business lever that maps to, and the total dollar change observed.

Write:
1. "action": ONE concrete, specific action a business owner could take (1-2 sentences).
   Ground it in the evidence packet's actual facts (categories, tickets, customer context).
2. "expected_impact": a FULL SENTENCE describing the potential impact, using ONLY the
   total_change_dollars figure already provided. Do NOT invent a new number or
   percentage that isn't derivable from the evidence packet. Format the dollar
   figure properly (e.g. "$1,592") and explain what it represents in context --
   for example: "If this action successfully reverses the volume decline, revenue
   could recover toward the observed baseline gap of approximately $1,592/week."
   NEVER respond with a bare number and nothing else.

STRICT RULES:
- Do not invent numbers, customers, or categories not present in the evidence packet.
- The action must be specific enough to assign to someone, not generic advice.
- expected_impact must be a complete, readable sentence, not a raw figure.

Respond ONLY with valid JSON in this exact shape:
{
  "action": "...",
  "expected_impact": "..."
}
"""


def get_lever(top_driver: str) -> dict:
    return LEVER_LOOKUP.get(top_driver, {"lever": "General review", "owner": "Analytics team"})


def build_monitoring_plan(kpi: str, region: str, segment: str, owner: str) -> str:
    """Deterministic template -- ties back to Detect's own re-run cadence
    rather than inventing new monitoring logic."""
    return (
        f"Re-run Detect on {kpi} for {region}/{segment} weekly for the next "
        f"4 weeks. Escalate to {owner} if the anomaly persists or worsens; "
        f"close out if the metric returns within the trailing baseline's "
        f"normal range."
    )


def recommend_for_flagged_scenario(explain_record: dict) -> dict:
    packet = explain_record["evidence_packet"]
    top_driver = packet["decomposition"]["top_driver"]
    lever_info = get_lever(top_driver)
    confidence = explain_record["forced_confidence_tier"]

    kpi, region, segment, week = packet["kpi"], packet["region"], packet["segment"], packet["week"]

    if confidence == "low_confidence_abstain":
        return {
            "kpi": kpi, "region": region, "segment": segment, "week": week,
            "driver": top_driver, "lever": lever_info["lever"], "owner": lever_info["owner"],
            "confidence": confidence,
            "action": "No specific action recommended -- evidence is insufficient to "
                      "confidently attribute a cause. Suggested first step: gather "
                      "supporting evidence (customer feedback, support tickets) for "
                      "the affected customers in this window before acting.",
            "expected_impact": "Not applicable -- no action is being recommended.",
            "monitoring_plan": build_monitoring_plan(kpi, region, segment, lever_info["owner"]),
            "telemetry": {"model_calls": 0, "total_latency_seconds": 0.0,
                          "total_tokens": 0, "total_estimated_cost_usd": 0.0},
        }

    total_change_dollars = packet["detected_change"]["value"] - packet["detected_change"]["baseline_median"]

    user_content = json.dumps({
        "evidence_packet": packet,
        "top_driver": top_driver,
        "lever": lever_info["lever"],
        "owner": lever_info["owner"],
        "total_change_dollars": round(total_change_dollars, 2),
    }, indent=2)

    messages = [
        {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    content, telemetry = call_groq(messages)
    llm_result = _parse_json_response(content)

    expected_impact = llm_result["expected_impact"]
    # Defensive fallback: if the model ignored the "full sentence" instruction
    # and returned a bare number, wrap it in a template sentence ourselves
    # rather than presenting an unexplained figure.
    if re.match(r"^[\s$\-\d,.]+$", expected_impact.strip()):
        expected_impact = (
            f"If this action addresses the {lever_info['lever'].lower()} driver, "
            f"revenue could move toward closing the observed baseline gap of "
            f"approximately ${abs(total_change_dollars):,.0f}/week for this slice."
        )

    return {
        "kpi": kpi, "region": region, "segment": segment, "week": week,
        "driver": top_driver, "lever": lever_info["lever"], "owner": lever_info["owner"],
        "confidence": confidence,
        "action": llm_result["action"],
        "expected_impact": expected_impact,
        "monitoring_plan": build_monitoring_plan(kpi, region, segment, lever_info["owner"]),
        "telemetry": {"model_calls": 1, "total_latency_seconds": telemetry["latency_seconds"],
                      "total_tokens": telemetry["prompt_tokens"] + telemetry["completion_tokens"],
                      "total_estimated_cost_usd": telemetry["estimated_cost_usd"]},
    }


def recommend_for_sparse_history_scenario(explain_record: dict) -> dict:
    packet = explain_record["evidence_packet"]
    kpi, region, segment, week = packet["kpi"], packet["region"], packet["segment"], packet["week"]
    return {
        "kpi": kpi, "region": region, "segment": segment, "week": week,
        "driver": "insufficient_history", "lever": "None -- baseline not yet established",
        "owner": "Analytics team", "confidence": "insufficient_history",
        "action": "No action recommended -- this is the first observed period for this "
                  "KPI/slice, with no baseline to judge whether the value is normal. "
                  "Suggested first step: continue collecting data for this slice; "
                  "revisit once at least 4 weeks of history are available.",
        "expected_impact": "Not applicable -- no action is being recommended.",
        "monitoring_plan": f"Continue weekly data collection for {kpi} on {region}/{segment}. "
                           f"Re-run Detect once at least 4 prior weeks of history exist.",
        "telemetry": {"model_calls": 0, "total_latency_seconds": 0.0,
                      "total_tokens": 0, "total_estimated_cost_usd": 0.0},
    }


def run_recommend(explain_results_path: str = "data/processed/explain_results.json") -> list:
    with open(explain_results_path) as f:
        explain_results = json.load(f)

    results = []
    for record in explain_results:
        if record["scenario_type"] == "sparse_history":
            rec = recommend_for_sparse_history_scenario(record)
        else:
            rec = recommend_for_flagged_scenario(record)
        results.append(rec)

    return results


if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY") is None:
        print("ERROR: GROQ_API_KEY not set (check your .env file).")
        raise SystemExit(1)

    print("Running Recommend on all Explain results...\n")
    results = run_recommend()

    grand_telemetry = {"model_calls": 0, "total_latency_seconds": 0.0, "total_tokens": 0, "total_estimated_cost_usd": 0.0}

    for r in results:
        print(f"{'='*70}\n{r['kpi']} {r['region']}/{r['segment']} {r['week']} -- confidence: {r['confidence']}\n{'='*70}")
        print(f"Driver: {r['driver']}  |  Lever: {r['lever']}  |  Owner: {r['owner']}")
        print(f"\nAction: {r['action']}")
        print(f"\nExpected impact: {r['expected_impact']}")
        print(f"\nMonitoring plan: {r['monitoring_plan']}")
        print(f"\nTelemetry: {r['telemetry']}\n")

        t = r["telemetry"]
        grand_telemetry["model_calls"] += t["model_calls"]
        grand_telemetry["total_latency_seconds"] += t["total_latency_seconds"]
        grand_telemetry["total_tokens"] += t["total_tokens"]
        grand_telemetry["total_estimated_cost_usd"] += t["total_estimated_cost_usd"]

    print(f"{'='*70}\nGRAND TOTAL TELEMETRY\n{'='*70}")
    print(json.dumps(grand_telemetry, indent=2))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/recommend_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWrote data/processed/recommend_results.json")
