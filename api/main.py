"""
Lens API + frontend host, in one FastAPI process.

Replaces the earlier stub endpoints with the REAL pipeline:
  /api/dashboard    -> reads real flagged anomalies from Detect's precomputed
                       output (data/processed/detect_results_flagged.csv),
                       filtered to the requesting persona's allowed regions.
  /api/investigate  -> governance.check_access() gates the request FIRST
                       (a denied request never touches the pipeline);
                       if allowed, runs the real Explain -> Recommend ->
                       Personas chain LIVE (real Groq calls), with a
                       persistent on-disk cache so repeat queries for the
                       same (persona, kpi, region, segment, week) don't
                       re-trigger paid API calls.
  /api/feedback     -> STILL A STUB. feedback.py doesn't exist yet; wiring
                       this properly is a separate, not-yet-done step.

Design note: what gets returned to the client
------------------------------------------------
Explain and Recommend reason over the FULL, unmasked evidence packet
internally (best-quality reasoning). Only personas.generate_persona_narrative()
applies masking, right before producing the final narrative. To make sure
PII can never leak to a client by accident, /api/investigate's response
NEVER includes the raw evidence_packet -- only the masked persona narrative
and the recommendation fields (driver/lever/action/confidence/monitoring_plan),
which contain no customer identifiers regardless of persona.

Cache design
----------------
Persistent, on-disk (data/processed/investigate_cache.json), so it survives
server restarts and doubles as a free record of every live query actually
run during development/demos. Keyed on (persona_id, kpi, region, segment,
week). A cache hit returns the ORIGINAL telemetry from when it was first
computed (not zeroed out) plus a "cache_hit": true flag, so the response is
honest about what it actually cost the first time without implying a new
cost was incurred on this request.
"""

import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import governance
from explain import explain_flagged_anomaly, explain_sparse_history
from recommend import recommend_for_flagged_scenario, recommend_for_sparse_history_scenario
from personas import generate_persona_narrative, PERSONAS

app = FastAPI(title="Lens API")

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "investigate_cache.json")
DETECT_FULL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "detect_results_full.csv")
DETECT_FLAGGED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "detect_results_flagged.csv")


# ---------------------------------------------------------------------------
# Persona resolution -- reuses governance.py's role table, does not duplicate it
# ---------------------------------------------------------------------------

def get_persona(x_persona_id: str = Header(default="finance_vp")):
    if x_persona_id not in governance.ROLE_ACCESS:
        raise HTTPException(status_code=401, detail=f"Unknown persona id '{x_persona_id}'")
    return x_persona_id


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache: dict):
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _cache_key(persona_id: str, kpi: str, region: str, segment: str, week: str) -> str:
    return f"{persona_id}|{kpi}|{region}|{segment}|{week}"


# ---------------------------------------------------------------------------
# Scenario lookup -- determines whether a (kpi,region,segment,week) query is
# a flagged anomaly, a sparse-history case, or doesn't exist / wasn't
# anomalous at all (distinct from an access denial).
# ---------------------------------------------------------------------------

def determine_scenario_type(kpi: str, region: str, segment: str, week: str) -> str:
    full = pd.read_csv(DETECT_FULL_PATH)
    full["week"] = pd.to_datetime(full["week"]).dt.date.astype(str)
    match = full[(full["kpi"] == kpi) & (full["region"] == region)
                 & (full["segment"] == segment) & (full["week"] == week)]
    if len(match) == 0:
        return "not_found"
    status = match.iloc[0]["status"]
    if status == "flagged":
        return "flagged_anomaly"
    elif status == "insufficient_history":
        return "sparse_history"
    else:
        return "not_anomalous"  # exists, but Detect never flagged it -- nothing to investigate


# ---------------------------------------------------------------------------
# /api/dashboard -- real flagged anomalies, filtered to the persona's allowed regions
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard(x_persona_id: str = Header(default="finance_vp")):
    persona_id = x_persona_id
    if persona_id not in governance.ROLE_ACCESS:
        raise HTTPException(status_code=401, detail=f"Unknown persona id '{persona_id}'")

    allowed_regions = governance.ROLE_ACCESS[persona_id]["allowed_regions"]

    flagged = pd.read_csv(DETECT_FLAGGED_PATH)
    flagged["week"] = pd.to_datetime(flagged["week"])
    if allowed_regions is not None:
        flagged = flagged[flagged["region"].isin(allowed_regions)]

    recent = flagged.sort_values("week", ascending=False).head(5)
    last_real_date = pd.to_datetime(flagged["week"]).max()

    kpis = []
    for _, row in recent.iterrows():
        kpis.append({
            "kpi": row["kpi"], "region": row["region"], "segment": row["segment"],
            "week": str(row["week"].date()),
            "value": round(float(row["value"]), 2),
            "delta_pct": round(float(row["pct_change"]), 1),
            "anomaly": True,
        })

    return {
        "as_of": str(last_real_date.date()) if pd.notna(last_real_date) else None,
        "freshness_note": "Historical demo data (real Superstore dataset, 2014-2017) -- "
                          "'as_of' reflects the dataset's real last date, not the current date.",
        "kpis": kpis,
    }


# ---------------------------------------------------------------------------
# /api/investigate -- the real Explain -> Recommend -> Personas chain, gated
# by governance, cached persistently.
# ---------------------------------------------------------------------------

class InvestigateRequest(BaseModel):
    kpi: str
    region: str
    segment: str
    week: str  # "YYYY-MM-DD"


@app.post("/api/investigate")
def investigate(req: InvestigateRequest, x_persona_id: str = Header(default="finance_vp")):
    persona_id = x_persona_id
    if persona_id not in governance.ROLE_ACCESS:
        raise HTTPException(status_code=401, detail=f"Unknown persona id '{persona_id}'")

    # 1. Access check FIRST -- before touching cache, before touching the pipeline.
    allowed, reason = governance.check_access(persona_id, req.region)
    governance.log_audit_entry({
        "persona_id": persona_id, "kpi": req.kpi, "region": req.region,
        "segment": req.segment, "week": req.week,
        "status": "allowed" if allowed else "denied", "reason": reason,
    })
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # 2. Cache check
    cache = _load_cache()
    key = _cache_key(persona_id, req.kpi, req.region, req.segment, req.week)
    if key in cache:
        result = dict(cache[key])
        result["cache_hit"] = True
        return result

    # 3. Determine scenario type using Detect's precomputed output
    scenario_type = determine_scenario_type(req.kpi, req.region, req.segment, req.week)
    if scenario_type == "not_found":
        raise HTTPException(status_code=404, detail="No data found for this KPI/region/segment/week.")
    if scenario_type == "not_anomalous":
        raise HTTPException(status_code=404, detail="This period was not flagged as anomalous -- nothing to investigate.")

    # 4. Run the REAL pipeline (live LLM calls happen here)
    if scenario_type == "flagged_anomaly":
        explain_record = explain_flagged_anomaly(req.kpi, req.region, req.segment, req.week)
        recommend_record = recommend_for_flagged_scenario(explain_record)
    else:  # sparse_history
        explain_record = explain_sparse_history(req.kpi, req.region, req.segment, req.week)
        recommend_record = recommend_for_sparse_history_scenario(explain_record)

    is_abstention = recommend_record["confidence"] in ("low_confidence_abstain", "insufficient_history")
    persona_key = governance.ROLE_TO_PERSONA_KEY[persona_id]

    if scenario_type == "flagged_anomaly":
        narrative, persona_telemetry = generate_persona_narrative(
            explain_record["evidence_packet"], recommend_record, persona_key, is_abstention
        )
        persona_call_telemetry = {
            "model_calls": 1, "latency_seconds": persona_telemetry["latency_seconds"],
            "tokens": persona_telemetry["prompt_tokens"] + persona_telemetry["completion_tokens"],
            "estimated_cost_usd": persona_telemetry["estimated_cost_usd"],
        }
    else:
        # Sparse-history already has a deterministic, non-LLM narrative from
        # explain_sparse_history -- no need for an extra persona LLM call to
        # rephrase "we have no data," which would just spend money to say
        # the same thing in slightly different words.
        narrative = explain_record["abstention_narrative"]
        persona_call_telemetry = {"model_calls": 0, "latency_seconds": 0.0, "tokens": 0, "estimated_cost_usd": 0.0}

    explain_t = explain_record.get("telemetry", {"model_calls": 0, "total_latency_seconds": 0.0,
                                                    "total_tokens": 0, "total_estimated_cost_usd": 0.0})
    recommend_t = recommend_record.get("telemetry", {"model_calls": 0, "total_latency_seconds": 0.0,
                                                        "total_tokens": 0, "total_estimated_cost_usd": 0.0})

    total_telemetry = {
        "model_calls": explain_t.get("model_calls", 0) + recommend_t.get("model_calls", 0) + persona_call_telemetry["model_calls"],
        "total_latency_seconds": round(
            explain_t.get("total_latency_seconds", 0.0) + recommend_t.get("total_latency_seconds", 0.0)
            + persona_call_telemetry["latency_seconds"], 3),
        "total_tokens": explain_t.get("total_tokens", 0) + recommend_t.get("total_tokens", 0) + persona_call_telemetry["tokens"],
        "total_estimated_cost_usd": round(
            explain_t.get("total_estimated_cost_usd", 0.0) + recommend_t.get("total_estimated_cost_usd", 0.0)
            + persona_call_telemetry["estimated_cost_usd"], 6),
    }

    result = {
        "status": "ok", "scenario_type": scenario_type,
        "kpi": req.kpi, "region": req.region, "segment": req.segment, "week": req.week,
        "confidence": recommend_record["confidence"],
        "driver": recommend_record["driver"], "lever": recommend_record["lever"],
        "owner": recommend_record["owner"],
        "action": recommend_record["action"], "expected_impact": recommend_record["expected_impact"],
        "monitoring_plan": recommend_record["monitoring_plan"],
        "narrative": narrative,
        "persona_view": PERSONAS[persona_key]["display_name"],
        "telemetry": total_telemetry,
        "cache_hit": False,
    }

    cache[key] = result
    _save_cache(cache)

    return result


# ---------------------------------------------------------------------------
# /api/feedback -- STILL A STUB. feedback.py does not exist yet.
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    kpi: str
    region: str
    segment: str
    week: str
    verdict: str  # "confirmed" | "rejected" | "edited"
    note: str | None = None


FEEDBACK_STORE = []  # TODO: replace with src.feedback persistent store once that module exists


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    FEEDBACK_STORE.append(req.dict())
    return {"status": "recorded", "total_feedback_entries": len(FEEDBACK_STORE)}


# ---------------------------------------------------------------------------
# Serve the static frontend at /
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
