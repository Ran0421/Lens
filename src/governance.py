"""
governance.py — RBAC / access-control gate for Lens

Non-LLM. This is the 4th required Round 2 demo scenario (role-based
security/entitlement), deliberately NOT built into explain.py or
personas.py -- see those files' docstrings for why access control is a
structurally different concern from reasoning (explain.py) or narrative
framing (personas.py).

Access is checked BEFORE any evidence packet is built -- not after, and
not just hidden from the response. A denied query never calls
build_evidence_packet() at all. This mirrors the architecture's stated
principle: "RBAC gates queries before data reaches the LLM (masking
happens pre-prompt)" -- here extended to: access gates queries before
data reaches ANYTHING, including the deterministic evidence-assembly step.

Every query -- allowed or denied -- is written to an append-only audit
log (data/processed/audit_log.jsonl), one JSON object per line, including
the denial reason. Nothing is ever deleted or overwritten from this file.

Roles (reusing, not duplicating, personas.py's masking logic)
------------------------------------------------------------------
  regional_manager_west / regional_manager_east:
    - allowed_regions: their own region only
    - mask_pii: False (can see account-level detail, matches
      personas.py's "regional_sales_manager" framing)

  finance_vp:
    - allowed_regions: all (no region restriction)
    - mask_pii: True (reuses personas.build_persona_view's existing
      finance_vp masking -- account-level identifiers stripped to
      aggregates, same as the narrative-framing stage)

Run:
    python3 src/governance.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(__file__))
from explain import build_evidence_packet
from personas import build_persona_view

AUDIT_LOG_PATH = "data/processed/audit_log.jsonl"

ROLE_ACCESS = {
    "regional_manager_west": {"allowed_regions": ["West"], "mask_pii": False},
    "regional_manager_east": {"allowed_regions": ["East"], "mask_pii": False},
    "finance_vp": {"allowed_regions": None, "mask_pii": True},  # None = all regions allowed
}

# Maps a governance role to the persona masking key already defined in
# personas.py, so masking logic is reused rather than reimplemented.
ROLE_TO_PERSONA_KEY = {
    "regional_manager_west": "regional_sales_manager",
    "regional_manager_east": "regional_sales_manager",
    "finance_vp": "finance_vp",
}


def check_access(persona_id: str, region: str) -> tuple:
    """Returns (allowed: bool, reason: str). Pure access check -- does not
    touch any data, just evaluates the role's permission against the
    requested region."""
    role = ROLE_ACCESS.get(persona_id)
    if role is None:
        return False, f"Unknown persona_id '{persona_id}'"

    allowed_regions = role["allowed_regions"]
    if allowed_regions is None:  # None means "all regions allowed"
        return True, "Role has all-region access"

    if region in allowed_regions:
        return True, f"Region '{region}' is within role's allowed regions {allowed_regions}"

    return False, f"Region '{region}' is outside role's allowed regions {allowed_regions}"


def log_audit_entry(entry: dict):
    """Appends one entry to the audit log. Append-only -- never opens the
    file in write/truncate mode, only append."""
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    entry_with_timestamp = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry_with_timestamp, default=str) + "\n")


def query_anomaly(persona_id: str, kpi: str, region: str, segment: str, week: str) -> dict:
    """The single entry point a caller (e.g. a future API layer) would use.
    Checks access FIRST; only builds an evidence packet if allowed."""
    allowed, reason = check_access(persona_id, region)

    base_log_entry = {
        "persona_id": persona_id, "kpi": kpi, "region": region,
        "segment": segment, "week": week, "status": "allowed" if allowed else "denied",
        "reason": reason,
    }

    if not allowed:
        log_audit_entry(base_log_entry)
        return {"status": "denied", "reason": reason, "evidence_packet": None}

    try:
        packet = build_evidence_packet(kpi, region, segment, week)
    except (IndexError, KeyError) as e:
        # The anomaly doesn't exist (e.g. wasn't flagged by Detect) -- this
        # is a "not found", distinct from an access denial, but still logged.
        log_audit_entry({**base_log_entry, "status": "not_found", "reason": str(e)})
        return {"status": "not_found", "reason": f"No flagged anomaly found for this query: {e}",
                "evidence_packet": None}

    persona_key = ROLE_TO_PERSONA_KEY[persona_id]
    masked_packet = build_persona_view(packet, persona_key)

    log_audit_entry(base_log_entry)
    return {"status": "allowed", "reason": reason, "evidence_packet": masked_packet,
            "masked": ROLE_ACCESS[persona_id]["mask_pii"]}


def run_demo_scenarios():
    """The 4 required governance demo cases, using real flagged anomalies
    already present in data/processed/ (no new LLM calls needed)."""
    scenarios = [
        {"label": "Regional Manager (West) queries their own region -- should ALLOW",
         "persona_id": "regional_manager_west", "kpi": "revenue",
         "region": "West", "segment": "Corporate", "week": "2017-10-30"},
        {"label": "Regional Manager (West) queries East -- should DENY",
         "persona_id": "regional_manager_west", "kpi": "revenue",
         "region": "East", "segment": "Corporate", "week": "2017-12-11"},
        {"label": "Finance VP queries East -- should ALLOW, masked",
         "persona_id": "finance_vp", "kpi": "revenue",
         "region": "East", "segment": "Corporate", "week": "2017-12-11"},
        {"label": "Finance VP queries West -- should ALLOW, masked (all-region access)",
         "persona_id": "finance_vp", "kpi": "revenue",
         "region": "West", "segment": "Corporate", "week": "2017-10-30"},
    ]

    for s in scenarios:
        print(f"\n{'='*70}\n{s['label']}\n{'='*70}")
        result = query_anomaly(s["persona_id"], s["kpi"], s["region"], s["segment"], s["week"])
        print(f"Status: {result['status']}")
        print(f"Reason: {result['reason']}")
        if result["status"] == "allowed":
            packet = result["evidence_packet"]
            print(f"Masked: {result['masked']}")
            print(f"customer_context in response: {packet.get('customer_context')}")
            print(f"supporting_tickets in response: {packet.get('supporting_tickets')}")

    print(f"\n{'='*70}\nAudit log ({AUDIT_LOG_PATH})\n{'='*70}")
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            print(f"  [{entry['timestamp']}] {entry['persona_id']} -> "
                  f"{entry['region']}/{entry['segment']} {entry['week']}: "
                  f"{entry['status']} ({entry['reason']})")


if __name__ == "__main__":
    run_demo_scenarios()
