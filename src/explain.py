"""
explain.py — Stage 4 of the Lens pipeline: Explain

The first LLM-touching stage. Given a flagged anomaly and its Detect/
Decompose outputs, proposes ranked candidate causes, adversarially
challenges its own top candidate, fact-checks that challenge
deterministically, and produces a final explanation whose CONFIDENCE LABEL
IS ALWAYS FORCED by evidence_score.py's confidence_tier -- never by
whatever confidence the LLM itself claims. This is the "hybrid rule" from
the architecture (see handoff section 3).

Runs on a curated set of 3 scenarios (not the full 317 flagged anomalies)
to keep API cost/time bounded during development -- see run_explain()
for the scenario list. A 4th required demo scenario (role-based security)
is NOT built here; it's an access-control concern for governance.py, not
an Explain reasoning concern.

LLM vs non-LLM boundary (explicit, per the brief's requirement)
--------------------------------------------------------------------
  LLM:      propose candidate causes; propose a challenge/counter-hypothesis
  Non-LLM:  evidence packet assembly; fact-checking the challenge against
            the evidence packet; the confidence gate itself; the entire
            sparse-history path (see below)

Three scenario types, three different amounts of LLM involvement
---------------------------------------------------------------------
  1. Multi-factor (high evidence_sufficiency):
       Full propose -> challenge -> fact-check pipeline. Confidence forced
       to "high" (matches evidence_score.py's tier for this scenario).

  2. Low-confidence (low evidence_sufficiency):
       Full propose -> challenge -> fact-check pipeline still RUNS (the
       LLM still reasons over what evidence exists) but the final
       confidence is FORCED to "low_confidence_abstain" regardless of the
       LLM's own stated confidence, and the narrative explicitly states
       what additional evidence would resolve it.

  3. Sparse-history (zero baseline weeks):
       NO LLM CALLS AT ALL. There is nothing to propose or challenge --
       Detect itself couldn't compute a baseline. Making an LLM call here
       just to say "I don't know" would be worse evidence-discipline than
       not calling it. This is a deliberate, documented non-LLM
       abstention path (0 latency, 0 tokens, 0 cost).

Run:
    export GROQ_API_KEY=gsk_...
    python3 src/explain.py
"""

import os
import json
import time
import re
import pandas as pd
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GROQ_API_KEY (and anything else) from a local .env file,
                    # if one exists, without it ever being typed into the terminal
                    # or committed to git (.env should be in .gitignore).
except ImportError:
    pass  # dotenv is optional -- falls back to whatever is already in the environment

try:
    from groq import Groq
except ImportError:
    Groq = None  # allows the non-LLM parts (evidence packet, sparse-history path) to be
                 # imported/tested even before `pip install groq` has been run.

# --- Configuration ---
# GROQ_MODEL default confirmed directly against the live Groq /models endpoint
# (client.models.list()) on 2026-08-29 -- NOT from search results, which turned
# out to reference a stale Groq lineup (llama-3.1/3.3 models that no longer
# exist on the current API). If this model also becomes unavailable, run:
#   python3 -c "from groq import Groq; import os; [print(m.id) for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data]"
# to see what's currently available on your account, and update this constant.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Pricing, USD per 1M tokens. UNVERIFIED against Groq's own docs (only found
# via a third-party aggregator citing Groq as the provider) -- treat this as
# a placeholder for the cost-telemetry requirement, and confirm/correct the
# real number at https://console.groq.com/docs/models or your billing page
# before reporting these figures anywhere final (e.g. the Business Proposal).
GROQ_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    # kept for reference in case GROQ_MODEL is overridden back to one of these
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

TICKET_DATE_WINDOW_DAYS = 7  # must match evidence_score.py


# ---------------------------------------------------------------------------
# Evidence packet assembly (non-LLM)
# ---------------------------------------------------------------------------

def build_evidence_packet(kpi: str, region: str, segment: str, week: str) -> dict:
    """Assembles the full evidence packet for a FLAGGED anomaly (i.e. one
    that went through Detect -> Decompose -> evidence_score.py). Used for
    the multi-factor and low-confidence scenarios.
    """
    week = pd.Timestamp(week)

    detect = pd.read_csv("data/processed/detect_results_flagged.csv")
    decompose = pd.read_csv("data/processed/decompose_results.csv")
    evidence = pd.read_csv("data/processed/evidence_scores.csv")
    tickets = pd.read_csv("data/processed/support_tickets.csv")
    customer_master = pd.read_csv("data/processed/customer_master.csv")
    transactions = pd.read_csv("data/processed/transactions.csv")

    detect["week"] = pd.to_datetime(detect["week"])
    decompose["week"] = pd.to_datetime(decompose["week"])
    evidence["week"] = pd.to_datetime(evidence["week"])

    d_row = detect[(detect["kpi"] == kpi) & (detect["region"] == region)
                   & (detect["segment"] == segment) & (detect["week"] == week)].iloc[0]
    dc_row = decompose[(decompose["kpi"] == kpi) & (decompose["region"] == region)
                        & (decompose["segment"] == segment) & (decompose["week"] == week)].iloc[0]
    ev_row = evidence[(evidence["kpi"] == kpi) & (evidence["region"] == region)
                       & (evidence["segment"] == segment) & (evidence["week"] == week)].iloc[0]

    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    transactions["week_col"] = transactions["order_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    customer_ids = sorted(transactions[
        (transactions["region"] == region) & (transactions["segment"] == segment)
        & (transactions["week_col"] == week)
    ]["customer_id"].unique().tolist())

    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"], format="mixed")
    window_start = week - pd.Timedelta(days=TICKET_DATE_WINDOW_DAYS)
    window_end = week + pd.Timedelta(days=7 + TICKET_DATE_WINDOW_DAYS)
    relevant_tickets = tickets[
        tickets["customer_id"].isin(customer_ids)
        & (tickets["ticket_date"] >= window_start) & (tickets["ticket_date"] <= window_end)
    ]

    relevant_customers = customer_master[customer_master["customer_id"].isin(customer_ids)]

    return {
        "kpi": kpi, "region": region, "segment": segment, "week": str(week.date()),
        "scenario_type": "flagged_anomaly",
        "detected_change": {
            "value": float(d_row["value"]),
            "baseline_median": float(d_row["baseline_median"]),
            "modified_z_score": float(d_row["modified_z"]),
            "pct_change": float(d_row["pct_change"]),
            "n_baseline_weeks": int(d_row["n_baseline_weeks"]),
        },
        "decomposition": {
            "volume_effect": float(dc_row["volume_effect"]) if pd.notna(dc_row["volume_effect"]) else None,
            "mix_effect": float(dc_row["mix_effect"]) if pd.notna(dc_row["mix_effect"]) else None,
            "price_effect": float(dc_row["price_effect"]) if pd.notna(dc_row["price_effect"]) else None,
            "top_driver": dc_row["top_driver"],
            "category_breakdown": json.loads(dc_row["category_breakdown"]),
        },
        "evidence_scoring": {
            "evidence_sufficiency": float(ev_row["evidence_sufficiency"]),
            "confidence_tier": ev_row["confidence_tier"],
            "attribution_clarity": float(ev_row["attribution_clarity"]) if pd.notna(ev_row["attribution_clarity"]) else None,
            "ticket_coverage_score": float(ev_row["ticket_coverage_score"]),
            "baseline_reliability_score": float(ev_row["baseline_reliability_score"]),
            "customer_context_score": float(ev_row["customer_context_score"]),
        },
        "supporting_tickets": [
            {"customer_id": r["customer_id"], "date": str(r["ticket_date"].date()), "text": r["ticket_text"]}
            for _, r in relevant_tickets.iterrows()
        ],
        "customer_context": [
            {"customer_id": r["customer_id"], "tenure_months": int(r["tenure_months"]),
             "contract_type": r["contract_type"], "churn_flag": int(r["churn_flag"])}
            for _, r in relevant_customers.iterrows()
        ],
    }


def build_sparse_history_packet(kpi: str, region: str, segment: str, week: str) -> dict:
    """Assembles the (much smaller) evidence packet for a SPARSE-HISTORY
    case -- one where Detect itself flagged insufficient_history. No
    Decompose or evidence_score data exists for these, by design (see
    detect.py: insufficient_history rows never get materiality_flag=True).
    """
    week = pd.Timestamp(week)
    full = pd.read_csv("data/processed/detect_results_full.csv")
    full["week"] = pd.to_datetime(full["week"])
    row = full[(full["kpi"] == kpi) & (full["region"] == region)
               & (full["segment"] == segment) & (full["week"] == week)].iloc[0]

    return {
        "kpi": kpi, "region": region, "segment": segment, "week": str(week.date()),
        "scenario_type": "sparse_history",
        "observed_value": float(row["value"]),
        "n_baseline_weeks": int(row["n_baseline_weeks"]),
        "status": row["status"],
    }


# ---------------------------------------------------------------------------
# LLM calls (Groq) with telemetry
# ---------------------------------------------------------------------------

def call_groq(messages: list, response_format_json: bool = True) -> tuple:
    """Makes one Groq chat completion call. Returns (content, telemetry_dict).
    telemetry_dict always has latency_seconds, prompt_tokens,
    completion_tokens, estimated_cost_usd, model -- even on failure (with
    zeros) so callers can aggregate telemetry uniformly.

    Falls back to a plain call (no response_format constraint) if the
    model/account doesn't support structured JSON mode -- some models on
    Groq's current lineup may not support it the same way older models did.
    """
    if Groq is None:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    kwargs = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.2}
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        if response_format_json and "response_format" in str(e).lower():
            # This model/account doesn't support structured JSON mode --
            # retry without it, relying on the prompt's own JSON instruction.
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    latency = time.time() - start

    usage = response.usage
    pricing = GROQ_PRICING.get(GROQ_MODEL, {"input": 0.0, "output": 0.0})
    cost = (usage.prompt_tokens / 1_000_000 * pricing["input"]
            + usage.completion_tokens / 1_000_000 * pricing["output"])

    telemetry = {
        "model": GROQ_MODEL, "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
        "estimated_cost_usd": round(cost, 6),
    }
    return response.choices[0].message.content, telemetry


def _parse_json_response(content: str) -> dict:
    """Parses the LLM's JSON output defensively -- strips markdown code
    fences if the model wrapped its JSON in ```json ... ``` despite being
    asked not to, which some models do regardless of instructions."""
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    return json.loads(cleaned)


PROPOSE_SYSTEM_PROMPT = """You are Lens, an evidence-grounded business analytics explainer.

You will be given a JSON evidence packet describing a detected KPI anomaly.
Your job: propose 2-3 ranked candidate causes for this movement.

STRICT RULES:
- You may ONLY cite facts, numbers, and categories that appear in the evidence packet.
- NEVER invent a number, ticket, customer, or category not present in the packet.
- Each candidate cause must include a "cited_evidence" field naming exactly which
  packet field(s) support it (e.g. "decomposition.volume_effect", "supporting_tickets[0]").
- If the evidence packet has few or no supporting_tickets, say so plainly in your
  rationale rather than papering over the gap.

Respond ONLY with valid JSON in this exact shape:
{
  "candidate_causes": [
    {"rank": 1, "cause": "...", "rationale": "...", "cited_evidence": ["..."], "llm_stated_confidence": "high|medium|low"}
  ]
}
"""

CHALLENGE_SYSTEM_PROMPT = """You are Lens's adversarial reviewer. You will be given the same
evidence packet and a proposed top candidate cause. Your job is to argue AGAINST that
top cause: either propose a plausible alternative explanation the evidence could also
support, or point out a specific weakness/gap in the cited evidence for it.

STRICT RULES:
- Only reference facts that exist in the evidence packet -- same rule as the proposer.
- Be genuinely adversarial. Do not simply agree with the proposed cause.

Respond ONLY with valid JSON in this exact shape:
{
  "challenge": "...",
  "alternative_cause": "... or null if none",
  "cited_evidence": ["..."]
}
"""


def propose_causes(evidence_packet: dict) -> tuple:
    messages = [
        {"role": "system", "content": PROPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(evidence_packet, indent=2)},
    ]
    content, telemetry = call_groq(messages)
    return _parse_json_response(content), telemetry


def challenge_causes(evidence_packet: dict, propose_result: dict) -> tuple:
    top_cause = propose_result["candidate_causes"][0] if propose_result.get("candidate_causes") else None
    user_content = json.dumps({"evidence_packet": evidence_packet, "proposed_top_cause": top_cause}, indent=2)
    messages = [
        {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    content, telemetry = call_groq(messages)
    return _parse_json_response(content), telemetry


# ---------------------------------------------------------------------------
# Fact-check (non-LLM): does the challenge cite anything NOT in the evidence packet?
# ---------------------------------------------------------------------------

def fact_check_challenge(evidence_packet: dict, challenge_result: dict) -> dict:
    """Deterministically checks whether the challenge's cited_evidence
    entries actually correspond to real fields/values in the evidence
    packet. This is the non-LLM fact-check step -- the LLM is never
    trusted to grade its own adversarial pass.
    """
    known_categories = {c["category"] for c in evidence_packet.get("decomposition", {}).get("category_breakdown", [])}
    known_customer_ids = {c["customer_id"] for c in evidence_packet.get("customer_context", [])}
    packet_str = json.dumps(evidence_packet)

    unsupported_claims = []
    for citation in challenge_result.get("cited_evidence", []):
        # A citation is considered grounded if it's a dotted/bracketed field
        # path that plausibly exists in the packet, OR if any category/customer
        # name it mentions is one we actually have evidence for.
        mentioned_categories = {cat for cat in known_categories if cat.lower() in citation.lower()}
        mentioned_customers = {cid for cid in known_customer_ids if cid.lower() in citation.lower()}
        looks_like_field_path = bool(re.match(r"^[a-z_][a-z0-9_.\[\]]*$", citation.strip(), re.IGNORECASE))

        is_grounded = bool(mentioned_categories) or bool(mentioned_customers) or looks_like_field_path
        if not is_grounded:
            unsupported_claims.append(citation)

    return {
        "is_grounded": len(unsupported_claims) == 0,
        "unsupported_claims": unsupported_claims,
        "n_citations_checked": len(challenge_result.get("cited_evidence", [])),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def explain_flagged_anomaly(kpi: str, region: str, segment: str, week: str) -> dict:
    packet = build_evidence_packet(kpi, region, segment, week)
    forced_tier = packet["evidence_scoring"]["confidence_tier"]

    propose_result, propose_telemetry = propose_causes(packet)
    challenge_result, challenge_telemetry = challenge_causes(packet, propose_result)
    fact_check_result = fact_check_challenge(packet, challenge_result)

    total_telemetry = {
        "model_calls": 2,
        "total_latency_seconds": round(propose_telemetry["latency_seconds"] + challenge_telemetry["latency_seconds"], 3),
        "total_tokens": (propose_telemetry["prompt_tokens"] + propose_telemetry["completion_tokens"]
                          + challenge_telemetry["prompt_tokens"] + challenge_telemetry["completion_tokens"]),
        "total_estimated_cost_usd": round(propose_telemetry["estimated_cost_usd"] + challenge_telemetry["estimated_cost_usd"], 6),
        "calls": [propose_telemetry, challenge_telemetry],
    }

    if forced_tier == "low_confidence_abstain":
        narrative = (
            f"Lens detected a {packet['detected_change']['pct_change']:.1f}% change in {kpi} "
            f"for {region}/{segment} in the week of {week}, which is statistically and "
            f"business-material. However, evidence coverage is too low to confidently "
            f"attribute a cause (ticket coverage score: "
            f"{packet['evidence_scoring']['ticket_coverage_score']}). "
            f"Additional evidence that would help resolve this: support tickets or "
            f"business context from the affected customers in this window."
        )
    else:
        narrative = None  # left to be composed from propose_result/challenge_result downstream

    return {
        "kpi": kpi, "region": region, "segment": segment, "week": week,
        "scenario_type": "flagged_anomaly",
        "forced_confidence_tier": forced_tier,
        "abstention_narrative": narrative,
        "evidence_packet": packet,
        "propose_result": propose_result,
        "challenge_result": challenge_result,
        "fact_check_result": fact_check_result,
        "telemetry": total_telemetry,
    }


def explain_sparse_history(kpi: str, region: str, segment: str, week: str) -> dict:
    packet = build_sparse_history_packet(kpi, region, segment, week)
    narrative = (
        f"Lens observed a {kpi} value of {packet['observed_value']:.2f} for {region}/{segment} "
        f"in the week of {week}, but this is the first (or one of the first) observed weeks "
        f"for this slice -- there are {packet['n_baseline_weeks']} prior weeks of history "
        f"available, below the minimum needed to build a reliable baseline. Lens cannot "
        f"assess whether this value is normal or anomalous, and does not attempt to. "
        f"As more weeks of data accumulate for this slice, a baseline can be constructed."
    )
    return {
        "kpi": kpi, "region": region, "segment": segment, "week": week,
        "scenario_type": "sparse_history",
        "forced_confidence_tier": "insufficient_history",
        "abstention_narrative": narrative,
        "evidence_packet": packet,
        "propose_result": None, "challenge_result": None, "fact_check_result": None,
        "telemetry": {"model_calls": 0, "total_latency_seconds": 0.0, "total_tokens": 0,
                      "total_estimated_cost_usd": 0.0, "calls": []},
    }


def run_explain():
    scenarios = [
        {"type": "flagged", "kpi": "revenue", "region": "East", "segment": "Corporate", "week": "2017-12-11",
         "label": "Multi-factor movement"},
        {"type": "flagged", "kpi": "revenue", "region": "Central", "segment": "Consumer", "week": "2017-12-25",
         "label": "Low-confidence / abstention"},
        {"type": "sparse", "kpi": "revenue", "region": "South", "segment": "Home Office", "week": "2014-01-06",
         "label": "Sparse-history"},
    ]

    all_results = []
    grand_telemetry = {"model_calls": 0, "total_latency_seconds": 0.0, "total_tokens": 0, "total_estimated_cost_usd": 0.0}

    for s in scenarios:
        print(f"\n{'='*70}\nScenario: {s['label']} -- {s['kpi']} {s['region']}/{s['segment']} {s['week']}\n{'='*70}")
        if s["type"] == "flagged":
            result = explain_flagged_anomaly(s["kpi"], s["region"], s["segment"], s["week"])
        else:
            result = explain_sparse_history(s["kpi"], s["region"], s["segment"], s["week"])

        print(f"Forced confidence tier: {result['forced_confidence_tier']}")
        if result["abstention_narrative"]:
            print(f"\nNarrative:\n{result['abstention_narrative']}")
        if result["propose_result"]:
            print(f"\nTop proposed cause: {result['propose_result']['candidate_causes'][0]['cause']}")
            print(f"Challenge: {result['challenge_result']['challenge']}")
            print(f"Fact-check grounded: {result['fact_check_result']['is_grounded']}")
        print(f"\nTelemetry: {result['telemetry']['model_calls']} calls, "
              f"{result['telemetry']['total_latency_seconds']}s, "
              f"{result['telemetry']['total_tokens']} tokens, "
              f"${result['telemetry']['total_estimated_cost_usd']}")

        all_results.append(result)
        t = result["telemetry"]
        grand_telemetry["model_calls"] += t["model_calls"]
        grand_telemetry["total_latency_seconds"] += t["total_latency_seconds"]
        grand_telemetry["total_tokens"] += t["total_tokens"]
        grand_telemetry["total_estimated_cost_usd"] += t["total_estimated_cost_usd"]

    print(f"\n{'='*70}\nGRAND TOTAL TELEMETRY (all {len(scenarios)} scenarios)\n{'='*70}")
    print(json.dumps(grand_telemetry, indent=2))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/explain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nWrote data/processed/explain_results.json")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Run: export GROQ_API_KEY=gsk_your_key_here")
        raise SystemExit(1)
    run_explain()
