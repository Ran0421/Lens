"""
llm_eval.py — Evaluation layer for Lens's LLM-touching stages

Non-LLM. Reads the already-generated outputs of explain.py, recommend.py,
and personas.py from data/processed/, and runs six deterministic checks
against them. No new API calls are made -- this is free and instant to
rerun any time those upstream outputs change.

Why these six checks specifically
-------------------------------------
This suite exists because we found a REAL bug this way, not hypothetically:
a Finance VP narrative correctly cited "$1,309.23" from the evidence packet
but appended a spurious "k" (thousands) suffix the model invented on its
own -- inflating the figure 1000x while the digits themselves were correct.
fact_check_challenge() (in explain.py) only checks whether a CITATION PATH
exists in the evidence packet; it is blind to whether the NUMBER quoted
alongside a real citation has been distorted. Check 5 below exists
specifically to catch this class of error.

1. citation_groundedness  -- extends explain.py's fact-check logic to
   BOTH the propose step's cited_evidence AND the challenge step's, across
   every flagged scenario. Reports an aggregate rate, not just pass/fail.

2. confidence_divergence  -- how often the LLM's own self-reported
   confidence (llm_stated_confidence on the top proposed cause) would have
   differed from the deterministic, evidence-based confidence_tier that
   actually gates the output. A high divergence rate is direct evidence
   for why the hybrid rule (forced confidence) matters.

3. pii_leak_scan          -- regex sweep of every finance_vp narrative for
   the Superstore customer-ID pattern. Should always find zero -- this
   checks the LLM's ACTUAL OUTPUT, not just the (already-verified) masked
   input packet.

4. entity_grounding       -- the inverse of #3: any customer ID mentioned
   in a regional_sales_manager narrative (which IS allowed to see them)
   must correspond to a real customer actually present in that scenario's
   evidence packet -- catches invented customers.

5. numeric_consistency    -- extracts dollar figures from every narrative/
   action/expected_impact text, strips any suffix multiplier (k/K/thousand/
   m/M/million), and checks whether the RAW digits match a real evidence
   packet field. If the raw digits match but a suffix was added (so the
   stated figure != the raw digits), flags a "distortion" -- this is the
   check that would have caught the k-suffix bug.

6. driver_alignment       -- does Explain's top proposed cause actually
   talk about the SAME driver category Decompose deterministically
   identified as top_driver? A keyword-overlap check, not semantic
   similarity -- simple and auditable.

Run:
    python3 src/llm_eval.py
"""

import json
import re
from pathlib import Path

# Superstore customer IDs are always 2 letters, hyphen, 5 digits (e.g. BP-11095)
CUSTOMER_ID_PATTERN = re.compile(r"\b[A-Z]{2}-\d{5}\b")

# Dollar figure with optional suffix multiplier, e.g. "$1,309.23", "$1,309.23 k", "$1.31 million"
DOLLAR_PATTERN = re.compile(
    r"\$\s*([\d,]+\.?\d*)\s*(k|K|thousand|m|M|million)?\b"
)

SUFFIX_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
}

DRIVER_KEYWORDS = {
    "volume_effect": ["volume", "order count", "orders", "quantity", "units"],
    "mix_effect": ["mix", "category shift", "assortment", "product mix"],
    "price_effect": ["price", "pricing", "discount"],
    "category_shift": ["category"],
}

NUMERIC_TOLERANCE_PCT = 0.02  # 2% tolerance for rounding differences


# ---------------------------------------------------------------------------
# 1. Citation groundedness
# ---------------------------------------------------------------------------

def _is_grounded_citation(citation: str, evidence_packet: dict) -> bool:
    known_categories = {c["category"] for c in evidence_packet.get("decomposition", {}).get("category_breakdown", [])}
    known_customer_ids = {c["customer_id"] for c in evidence_packet.get("customer_context", [])}
    mentioned_categories = {cat for cat in known_categories if cat.lower() in citation.lower()}
    mentioned_customers = {cid for cid in known_customer_ids if cid.lower() in citation.lower()}
    looks_like_field_path = bool(re.match(r"^[a-z_][a-z0-9_.\[\]]*$", citation.strip(), re.IGNORECASE))
    return bool(mentioned_categories) or bool(mentioned_customers) or looks_like_field_path


def check_citation_groundedness(explain_results: list) -> dict:
    total_citations, unsupported = 0, []
    for record in explain_results:
        if record["scenario_type"] != "flagged_anomaly":
            continue
        packet = record["evidence_packet"]
        scenario_key = f"{packet['kpi']} {packet['region']}/{packet['segment']} {packet['week']}"

        for cause in record.get("propose_result", {}).get("candidate_causes", []) or []:
            for citation in cause.get("cited_evidence", []):
                total_citations += 1
                if not _is_grounded_citation(citation, packet):
                    unsupported.append({"scenario": scenario_key, "stage": "propose", "citation": citation})

        for citation in (record.get("challenge_result") or {}).get("cited_evidence", []):
            total_citations += 1
            if not _is_grounded_citation(citation, packet):
                unsupported.append({"scenario": scenario_key, "stage": "challenge", "citation": citation})

    rate = 1.0 - (len(unsupported) / total_citations) if total_citations else None
    return {"total_citations_checked": total_citations, "unsupported_citations": unsupported,
            "groundedness_rate": round(rate, 3) if rate is not None else None}


# ---------------------------------------------------------------------------
# 2. Confidence divergence
# ---------------------------------------------------------------------------

def check_confidence_divergence(explain_results: list) -> dict:
    divergences = []
    for record in explain_results:
        if record["scenario_type"] != "flagged_anomaly":
            continue
        causes = record.get("propose_result", {}).get("candidate_causes", []) or []
        if not causes:
            continue
        llm_confidence = causes[0].get("llm_stated_confidence", "").lower()
        forced_tier = record["forced_confidence_tier"]
        forced_simplified = "low" if forced_tier == "low_confidence_abstain" else forced_tier

        packet = record["evidence_packet"]
        scenario_key = f"{packet['kpi']} {packet['region']}/{packet['segment']} {packet['week']}"

        diverged = llm_confidence != forced_simplified
        divergences.append({
            "scenario": scenario_key, "llm_stated_confidence": llm_confidence,
            "forced_confidence_tier": forced_tier, "diverged": diverged,
        })

    n_diverged = sum(1 for d in divergences if d["diverged"])
    rate = n_diverged / len(divergences) if divergences else None
    return {"comparisons": divergences, "n_diverged": n_diverged,
            "divergence_rate": round(rate, 3) if rate is not None else None}


# ---------------------------------------------------------------------------
# 3 & 4. PII leak scan / entity grounding
# ---------------------------------------------------------------------------

def check_pii_and_entity_grounding(personas_results: list, explain_results: list) -> dict:
    packets_by_key = {}
    for record in explain_results:
        packet = record["evidence_packet"]
        key = f"{packet['kpi']} {packet['region']}/{packet['segment']} {packet['week']}"
        packets_by_key[key] = packet

    pii_leaks = []
    ungrounded_entities = []

    for scenario in personas_results:
        key = f"{scenario['kpi']} {scenario['region']}/{scenario['segment']} {scenario['week']}"
        packet = packets_by_key.get(key, {})
        known_customer_ids = {c["customer_id"] for c in packet.get("customer_context", [])}

        fvp_narrative = scenario["narratives"].get("finance_vp", {}).get("narrative", "")
        found_in_fvp = CUSTOMER_ID_PATTERN.findall(fvp_narrative)
        if found_in_fvp:
            pii_leaks.append({"scenario": key, "leaked_ids": found_in_fvp})

        rsm_narrative = scenario["narratives"].get("regional_sales_manager", {}).get("narrative", "")
        found_in_rsm = set(CUSTOMER_ID_PATTERN.findall(rsm_narrative))
        ungrounded = found_in_rsm - known_customer_ids
        if ungrounded:
            ungrounded_entities.append({"scenario": key, "invented_ids": sorted(ungrounded)})

    return {
        "pii_leaks": pii_leaks, "n_pii_leaks": len(pii_leaks),
        "ungrounded_entities": ungrounded_entities, "n_ungrounded_entities": len(ungrounded_entities),
    }


# ---------------------------------------------------------------------------
# 5. Numeric consistency
# ---------------------------------------------------------------------------

def _collect_numeric_fields(evidence_packet: dict) -> list:
    """Every numeric value in the evidence packet that a narrative might
    legitimately quote, as (field_name, absolute_value) pairs."""
    fields = []
    dc = evidence_packet.get("detected_change", {})
    for k in ("value", "baseline_median"):
        if k in dc and dc[k] is not None:
            fields.append((f"detected_change.{k}", abs(dc[k])))

    # Derived: recommend.py computes total_change_dollars = value - baseline_median.
    # This is a legitimate, real derived figure (not a hallucination) even though
    # it isn't itself a raw packet field -- so it belongs in the known-fields set.
    if "value" in dc and "baseline_median" in dc and dc["value"] is not None and dc["baseline_median"] is not None:
        fields.append(("detected_change.value_minus_baseline_median", abs(dc["value"] - dc["baseline_median"])))

    dec = evidence_packet.get("decomposition", {})
    for k in ("volume_effect", "mix_effect", "price_effect"):
        if dec.get(k) is not None:
            fields.append((f"decomposition.{k}", abs(dec[k])))

    for cat in dec.get("category_breakdown", []) or []:
        for k in ("current_value", "baseline_value", "delta"):
            if cat.get(k) is not None:
                fields.append((f"category[{cat.get('category')}].{k}", abs(cat[k])))

    return fields


def _matches_any(value: float, known_fields: list, tolerance_pct: float) -> str:
    for name, known_value in known_fields:
        if known_value == 0:
            continue
        if abs(value - known_value) / known_value <= tolerance_pct:
            return name
    return None


def check_numeric_consistency(explain_results: list, recommend_results: list, personas_results: list) -> dict:
    packets_by_key = {}
    for record in explain_results:
        packet = record["evidence_packet"]
        key = f"{packet['kpi']} {packet['region']}/{packet['segment']} {packet['week']}"
        packets_by_key[key] = packet

    distortions = []
    unverifiable = []
    total_checked = 0

    text_sources = []
    for r in recommend_results:
        key = f"{r['kpi']} {r['region']}/{r['segment']} {r['week']}"
        text_sources.append((key, "recommend.action", r.get("action", "")))
        text_sources.append((key, "recommend.expected_impact", r.get("expected_impact", "")))
    for s in personas_results:
        key = f"{s['kpi']} {s['region']}/{s['segment']} {s['week']}"
        for persona_key, data in s["narratives"].items():
            text_sources.append((key, f"personas.{persona_key}", data.get("narrative", "")))

    for key, source, text in text_sources:
        packet = packets_by_key.get(key)
        if not packet:
            continue
        known_fields = _collect_numeric_fields(packet)

        for match in DOLLAR_PATTERN.finditer(text):
            raw_str, suffix = match.groups()
            raw_value = float(raw_str.replace(",", ""))
            total_checked += 1

            multiplier = SUFFIX_MULTIPLIERS.get((suffix or "").lower(), 1)
            stated_value = raw_value * multiplier

            raw_match = _matches_any(raw_value, known_fields, NUMERIC_TOLERANCE_PCT)
            stated_match = _matches_any(stated_value, known_fields, NUMERIC_TOLERANCE_PCT)

            if multiplier != 1 and raw_match and not stated_match:
                distortions.append({
                    "scenario": key, "source": source, "quoted_text": match.group(0),
                    "raw_digits_match_field": raw_match, "distorted_stated_value": stated_value,
                })
            elif not raw_match and not stated_match:
                unverifiable.append({
                    "scenario": key, "source": source, "quoted_text": match.group(0),
                })

    return {
        "total_dollar_figures_checked": total_checked,
        "distortions": distortions, "n_distortions": len(distortions),
        "unverifiable_figures": unverifiable, "n_unverifiable": len(unverifiable),
    }


# ---------------------------------------------------------------------------
# 6. Driver alignment
# ---------------------------------------------------------------------------

def check_driver_alignment(explain_results: list) -> dict:
    results = []
    for record in explain_results:
        if record["scenario_type"] != "flagged_anomaly":
            continue
        packet = record["evidence_packet"]
        key = f"{packet['kpi']} {packet['region']}/{packet['segment']} {packet['week']}"
        top_driver = packet["decomposition"]["top_driver"]
        keywords = DRIVER_KEYWORDS.get(top_driver, [])

        causes = record.get("propose_result", {}).get("candidate_causes", []) or []
        if not causes:
            continue
        top_cause_text = (causes[0].get("cause", "") + " " + causes[0].get("rationale", "")).lower()

        aligned = any(kw in top_cause_text for kw in keywords)
        results.append({"scenario": key, "top_driver": top_driver, "aligned": aligned})

    n_aligned = sum(1 for r in results if r["aligned"])
    rate = n_aligned / len(results) if results else None
    return {"comparisons": results, "n_aligned": n_aligned,
            "alignment_rate": round(rate, 3) if rate is not None else None}


# ---------------------------------------------------------------------------
# Telemetry rollup
# ---------------------------------------------------------------------------

def rollup_telemetry(explain_results: list, recommend_results: list, personas_results: list) -> dict:
    total = {"model_calls": 0, "total_latency_seconds": 0.0, "total_tokens": 0, "total_estimated_cost_usd": 0.0}

    for r in explain_results:
        t = r.get("telemetry", {})
        total["model_calls"] += t.get("model_calls", 0)
        total["total_latency_seconds"] += t.get("total_latency_seconds", 0.0)
        total["total_tokens"] += t.get("total_tokens", 0)
        total["total_estimated_cost_usd"] += t.get("total_estimated_cost_usd", 0.0)

    for r in recommend_results:
        t = r.get("telemetry", {})
        total["model_calls"] += t.get("model_calls", 0)
        total["total_latency_seconds"] += t.get("total_latency_seconds", 0.0)
        total["total_tokens"] += t.get("total_tokens", 0)
        total["total_estimated_cost_usd"] += t.get("total_estimated_cost_usd", 0.0)

    for s in personas_results:
        for persona_key, data in s["narratives"].items():
            t = data.get("telemetry", {})
            total["model_calls"] += t.get("model_calls", 0)
            total["total_latency_seconds"] += t.get("latency_seconds", 0.0)
            total["total_tokens"] += t.get("tokens", 0)
            total["total_estimated_cost_usd"] += t.get("estimated_cost_usd", 0.0)

    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in total.items()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eval(
    explain_path: str = "data/processed/explain_results.json",
    recommend_path: str = "data/processed/recommend_results.json",
    personas_path: str = "data/processed/personas_results.json",
) -> dict:
    with open(explain_path) as f:
        explain_results = json.load(f)
    with open(recommend_path) as f:
        recommend_results = json.load(f)
    with open(personas_path) as f:
        personas_results = json.load(f)

    return {
        "citation_groundedness": check_citation_groundedness(explain_results),
        "confidence_divergence": check_confidence_divergence(explain_results),
        "pii_and_entity_grounding": check_pii_and_entity_grounding(personas_results, explain_results),
        "numeric_consistency": check_numeric_consistency(explain_results, recommend_results, personas_results),
        "driver_alignment": check_driver_alignment(explain_results),
        "telemetry_rollup": rollup_telemetry(explain_results, recommend_results, personas_results),
    }


if __name__ == "__main__":
    print("Running LLM evaluation suite (no new API calls -- reads existing outputs)...\n")
    report = run_eval()

    print("=== 1. Citation groundedness ===")
    print(f"Rate: {report['citation_groundedness']['groundedness_rate']} "
          f"({report['citation_groundedness']['total_citations_checked']} citations checked)")
    if report["citation_groundedness"]["unsupported_citations"]:
        print("Unsupported citations found:", report["citation_groundedness"]["unsupported_citations"])

    print("\n=== 2. Confidence divergence ===")
    print(f"Rate: {report['confidence_divergence']['divergence_rate']} "
          f"({report['confidence_divergence']['n_diverged']} diverged)")
    for c in report["confidence_divergence"]["comparisons"]:
        print(f"  {c['scenario']}: LLM said '{c['llm_stated_confidence']}', "
              f"forced to '{c['forced_confidence_tier']}' -> diverged={c['diverged']}")

    print("\n=== 3 & 4. PII leaks / entity grounding ===")
    pg = report["pii_and_entity_grounding"]
    print(f"PII leaks in Finance VP narratives: {pg['n_pii_leaks']}")
    print(f"Invented entities in Sales Manager narratives: {pg['n_ungrounded_entities']}")

    print("\n=== 5. Numeric consistency ===")
    nc = report["numeric_consistency"]
    print(f"Dollar figures checked: {nc['total_dollar_figures_checked']}")
    print(f"Distortions found (suffix inflation etc): {nc['n_distortions']}")
    for d in nc["distortions"]:
        print(f"  [{d['scenario']}] {d['source']}: quoted '{d['quoted_text']}' -- "
              f"raw digits matched {d['raw_digits_match_field']}, but stated value "
              f"({d['distorted_stated_value']:.2f}) does not match any evidence field")
    print(f"Unverifiable figures: {nc['n_unverifiable']}")

    print("\n=== 6. Driver alignment ===")
    da = report["driver_alignment"]
    print(f"Rate: {da['alignment_rate']} ({da['n_aligned']}/{len(da['comparisons'])} aligned)")

    print("\n=== Telemetry rollup (all stages) ===")
    print(json.dumps(report["telemetry_rollup"], indent=2))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/llm_eval_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWrote data/processed/llm_eval_report.json")
