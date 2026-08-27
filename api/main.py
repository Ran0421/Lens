"""
Lens API + frontend host, in one FastAPI process.

Designed to run as-is on Replit: one process serves the API under /api/*
and the static frontend at /. This means the whole prototype is reachable
from a single public Replit URL -- useful for the demo video and for
anyone reviewing the GitHub repo without needing to run two servers.

Endpoints below are stubbed with realistic shapes so the frontend can be
built and demoed against them now; each stub is replaced with the real
Detect/Decompose/Explain/Recommend pipeline as those modules land (see
README status checklist). Stubs are clearly marked -- nothing here
pretends to be more finished than it is.
"""

import os
import time
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Lens API")

# ---------------------------------------------------------------------------
# Minimal RBAC stub -- real version reads from a permissions table (governance.py)
# ---------------------------------------------------------------------------
PERSONA_ACCESS = {
    "regional_manager_west": {"role": "regional_manager", "region": "West"},
    "regional_manager_east": {"role": "regional_manager", "region": "East"},
    "finance_vp": {"role": "finance_vp", "region": None},  # all regions
}


def get_persona(x_persona_id: str = Header(default="finance_vp")):
    persona = PERSONA_ACCESS.get(x_persona_id)
    if not persona:
        raise HTTPException(status_code=401, detail="Unknown persona id")
    return {"id": x_persona_id, **persona}


# ---------------------------------------------------------------------------
# /api/dashboard -- summary KPI tiles
# ---------------------------------------------------------------------------
@app.get("/api/dashboard")
def dashboard(persona=None):
    # STUB: replace with real query against reconcile.Warehouse once Detect lands
    return {
        "as_of": "2026-08-22",
        "kpis": [
            {"kpi": "revenue", "value": 4820000, "delta_pct": 3.4, "anomaly": False},
            {"kpi": "revenue_northeast", "value": 912000, "delta_pct": -8.0, "anomaly": True},
            {"kpi": "repeat_purchase_rate", "value": 0.34, "delta_pct": -0.3, "anomaly": False,
             "freshness_note": "weekly snapshot, last refreshed 4 days ago"},
        ],
    }


# ---------------------------------------------------------------------------
# /api/investigate -- runs Detect(already flagged) -> Decompose -> Explain -> Recommend
# ---------------------------------------------------------------------------
class InvestigateRequest(BaseModel):
    kpi: str
    scenario: str = "clear"  # "clear" | "ambiguous" | "sparse_history" | "multi_factor"


@app.post("/api/investigate")
def investigate(req: InvestigateRequest, persona=None):
    start = time.time()

    # STUB: real version calls src.decompose, src.explain, src.recommend
    # and returns their actual output. Shape below is the contract the
    # frontend is built against.
    result = {
        "localize": {
            "summary": "Most of the decline is concentrated in Northeast / Enterprise accounts.",
            "breakdown": [
                {"slice": "Northeast / Enterprise", "impact_pp": -7.2},
                {"slice": "Northeast / Mid-market", "impact_pp": -1.1},
            ],
        },
        "causes": [
            {"name": "Pricing complaints spiked in Northeast enterprise accounts",
             "confidence": "high", "evidence_refs": ["ticket_batch_2026w33", "call_notes_2026w33"]},
        ],
        "confidence_level": "high" if req.scenario == "clear" else "low",
        "recommendations": (
            ["Schedule account review for 9 flagged Northeast enterprise accounts "
             "($612K annual revenue at risk)"]
            if req.scenario == "clear" else []
        ),
        "data_gap": (
            None if req.scenario == "clear"
            else "Ticket coverage for this segment is incomplete this period -- "
                 "recommend pulling win/loss call notes before trusting a cause."
        ),
        "telemetry": {
            "latency_ms": round((time.time() - start) * 1000, 1),
            "llm_calls": 1,
            "tokens_estimated": 850,
            "cost_estimated_usd": 0.0004,
            "stages": {"reconcile": "non-llm", "decompose": "non-llm",
                       "explain": "llm (evidence-gated)", "recommend": "hybrid"},
        },
    }
    return result


# ---------------------------------------------------------------------------
# /api/feedback -- capture + (lightweight) closed loop
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    investigation_id: str
    cause_name: str
    verdict: str  # "confirmed" | "rejected" | "edited"
    note: str | None = None


FEEDBACK_STORE = []  # STUB: replace with src.feedback persistent store


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    FEEDBACK_STORE.append(req.dict())
    return {"status": "recorded", "total_feedback_entries": len(FEEDBACK_STORE)}


# ---------------------------------------------------------------------------
# Serve the static frontend at /
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
