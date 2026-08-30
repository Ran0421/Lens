"""
personas.py — Stage 6 of the Lens pipeline: Personas

LLM stage, but split the same way as every prior stage: MASKING is
deterministic (non-LLM), NARRATIVE PHRASING is LLM.

Two personas (per the handoff, already locked):
  - regional_sales_manager: tactical, account-level detail, operational tone
  - finance_vp: aggregated, forward-risk framing, NO account-level PII

Why masking happens before the LLM call, not as a prompt instruction
------------------------------------------------------------------------
This mirrors the architecture's stated non-functional design: "RBAC gates
queries before data reaches the LLM (masking happens pre-prompt)." For
finance_vp, customer_id and raw ticket text are stripped out of the
evidence packet BEFORE it's ever included in a prompt -- it's not that the
LLM is asked to avoid mentioning them, it's that the information is
structurally absent from what it's given. This makes a PII leak into the
Finance VP narrative a data-flow impossibility, not just an unlikely
prompting failure.

Note on scope vs governance.py
-----------------------------------
personas.py assumes the caller ALREADY has legitimate access to the
underlying anomaly (that gate is governance.py's job -- row/region-level
access control, not built yet). personas.py's only job is: given
legitimate access, how much DETAIL and what FRAMING does this persona see.
A Regional Sales Manager being restricted to their OWN region is an access
question (governance.py), not a framing question (this file).

Abstention scenarios (low-confidence, sparse-history) still get persona
narratives, but the LLM's job is narrower there: REPHRASE the same
abstention decision for the right audience, not generate any new causal
claim. The prompt explicitly forbids inventing a cause -- Explain/Recommend
already decided to abstain, and personas.py must not undo that by having a
persona-flavored narrative sound more confident than the underlying finding.

Run:
    export GROQ_API_KEY=gsk_...  (or set in .env)
    python3 src/personas.py
"""

import os
import json
import copy
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
sys.path.insert(0, os.path.dirname(__file__))
from explain import call_groq, _parse_json_response


PERSONAS = {
    "regional_sales_manager": {
        "display_name": "Regional Sales Manager",
        "mask_pii": False,
        "tone_instruction": (
            "Tactical and operational. Write for someone managing day-to-day "
            "accounts in this region. Use account-level specifics (customer "
            "names/IDs, specific tickets) when available -- this persona is "
            "authorized to see them. Be direct and action-oriented: what "
            "should they do this week."
        ),
    },
    "finance_vp": {
        "display_name": "Finance VP",
        "mask_pii": True,
        "tone_instruction": (
            "Aggregated and strategic. Write for a financial executive "
            "reviewing this across all regions, not one account. NEVER "
            "mention individual customer names/IDs -- speak in terms of "
            "counts, categories, and dollar magnitude only. Emphasize "
            "forward-looking risk and whether this needs escalation, not "
            "day-to-day tactics."
        ),
    },
}


def build_persona_view(evidence_packet: dict, persona_key: str) -> dict:
    """Deterministically masks the evidence packet for the given persona.
    See module docstring: this happens BEFORE any LLM call, not as a prompt
    instruction."""
    packet = copy.deepcopy(evidence_packet)
    persona = PERSONAS[persona_key]

    if not persona["mask_pii"]:
        return packet  # regional_sales_manager sees the full packet, unmasked

    # finance_vp: strip customer-level identifiers, replace with aggregates
    if "supporting_tickets" in packet:
        n_tickets = len(packet["supporting_tickets"])
        packet["supporting_tickets"] = {
            "n_tickets": n_tickets,
            "note": "Individual ticket text and customer identifiers are masked for this persona.",
        }

    if "customer_context" in packet:
        customers = packet["customer_context"]
        n = len(customers)
        avg_tenure = round(sum(c["tenure_months"] for c in customers) / n, 1) if n else None
        churn_rate = round(sum(c["churn_flag"] for c in customers) / n, 2) if n else None
        packet["customer_context"] = {
            "n_customers": n, "avg_tenure_months": avg_tenure, "churn_rate": churn_rate,
            "note": "Individual customer identifiers are masked for this persona.",
        }

    return packet


NARRATIVE_SYSTEM_PROMPT = """You are Lens, writing a persona-specific narrative summarizing
a business finding for a specific audience.

Audience: {display_name}
Tone/framing instructions: {tone_instruction}

You will be given: a (possibly masked) evidence packet, the recommendation's driver/lever/
action/confidence, and whether this is an ABSTENTION case (no cause is being claimed).

STRICT RULES:
- Only reference facts present in the evidence packet you were given. If customer
  identifiers are masked, do not mention or infer specific customer names.
- If this is an ABSTENTION case (is_abstention=true), your narrative must NOT claim or
  imply a specific cause. Rephrase the abstention for this audience -- e.g. what it means
  for them, whether it needs attention -- without inventing an explanation.
- Keep it to one short paragraph (3-5 sentences).

Respond ONLY with valid JSON in this exact shape:
{{
  "narrative": "..."
}}
"""


def generate_persona_narrative(evidence_packet: dict, recommend_result: dict,
                                 persona_key: str, is_abstention: bool) -> tuple:
    persona = PERSONAS[persona_key]
    masked_packet = build_persona_view(evidence_packet, persona_key)

    system_prompt = NARRATIVE_SYSTEM_PROMPT.format(
        display_name=persona["display_name"], tone_instruction=persona["tone_instruction"]
    )
    user_content = json.dumps({
        "evidence_packet": masked_packet,
        "driver": recommend_result["driver"], "lever": recommend_result["lever"],
        "action": recommend_result["action"], "confidence": recommend_result["confidence"],
        "is_abstention": is_abstention,
    }, indent=2)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    content, telemetry = call_groq(messages)
    result = _parse_json_response(content)
    return result["narrative"], telemetry


def run_personas(
    explain_results_path: str = "data/processed/explain_results.json",
    recommend_results_path: str = "data/processed/recommend_results.json",
) -> list:
    with open(explain_results_path) as f:
        explain_results = json.load(f)
    with open(recommend_results_path) as f:
        recommend_results = json.load(f)

    results = []
    for explain_record, recommend_record in zip(explain_results, recommend_results):
        packet = explain_record["evidence_packet"]
        is_abstention = recommend_record["confidence"] in ("low_confidence_abstain", "insufficient_history")

        scenario_result = {
            "kpi": packet["kpi"], "region": packet["region"], "segment": packet["segment"],
            "week": packet["week"], "confidence": recommend_record["confidence"],
            "is_abstention": is_abstention, "narratives": {},
        }

        for persona_key in PERSONAS:
            narrative, telemetry = generate_persona_narrative(
                packet, recommend_record, persona_key, is_abstention
            )
            scenario_result["narratives"][persona_key] = {
                "narrative": narrative,
                "telemetry": {"model_calls": 1, "latency_seconds": telemetry["latency_seconds"],
                              "tokens": telemetry["prompt_tokens"] + telemetry["completion_tokens"],
                              "estimated_cost_usd": telemetry["estimated_cost_usd"]},
            }

        results.append(scenario_result)

    return results


if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY") is None:
        print("ERROR: GROQ_API_KEY not set (check your .env file).")
        raise SystemExit(1)

    print("Running Personas on all Explain/Recommend results...\n")
    results = run_personas()

    grand_telemetry = {"model_calls": 0, "total_latency_seconds": 0.0, "total_tokens": 0, "total_estimated_cost_usd": 0.0}

    for r in results:
        print(f"{'='*70}\n{r['kpi']} {r['region']}/{r['segment']} {r['week']} "
              f"-- confidence: {r['confidence']} (abstention: {r['is_abstention']})\n{'='*70}")
        for persona_key, data in r["narratives"].items():
            print(f"\n--- {PERSONAS[persona_key]['display_name']} ---")
            print(data["narrative"])
            t = data["telemetry"]
            grand_telemetry["model_calls"] += t["model_calls"]
            grand_telemetry["total_latency_seconds"] += t["latency_seconds"]
            grand_telemetry["total_tokens"] += t["tokens"]
            grand_telemetry["total_estimated_cost_usd"] += t["estimated_cost_usd"]
        print()

    print(f"{'='*70}\nGRAND TOTAL TELEMETRY\n{'='*70}")
    print(json.dumps(grand_telemetry, indent=2))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/personas_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWrote data/processed/personas_results.json")
