"""
evidence_score.py — Deterministic evidence scoring for the Explain stage

Non-LLM. Computes two SEPARATE signals for every flagged anomaly, on
purpose kept apart (see rationale below):

1. evidence_sufficiency  -- gates whether Explain is allowed to express
   confidence at all. Weakest-link minimum across three evidentiary
   dimensions: ticket_coverage, baseline_reliability, customer_context.
   This is the "hybrid rule" from the architecture: it overrides whatever
   confidence the LLM itself might claim.

2. attribution_clarity (revenue anomalies only) -- does NOT affect the
   confidence gate. It only shapes how the narrative is PHRASED: whether
   Explain should name one dominant driver, or say "multiple factors
   contributed roughly equally."

Why these are two separate scores, not one combined score
------------------------------------------------------------
An earlier version of this design combined decomposition-dominance into
the same weakest-link score as the evidentiary dimensions. That was wrong:
a multi-factor anomaly where volume/mix/price all contributed comparably
is NOT evidence of uncertainty -- the reconciliation error is exactly 0,
we know precisely how much each factor contributed. What's "unclear" there
is only the NARRATIVE (hard to name one villain), not the EVIDENCE. Gating
confidence on decomposition-dominance would have mislabeled genuinely
well-evidenced multi-factor anomalies as low-confidence, directly
undermining the requirement to demonstrate a multi-factor scenario as
DISTINCT from a low-confidence scenario. Splitting the two signals fixes
this: a multi-factor anomaly can correctly be (high evidence_sufficiency,
low attribution_clarity), while a genuinely under-evidenced anomaly is
(low evidence_sufficiency, whatever attribution_clarity happens to be).

Scoring formulas
-------------------
ticket_coverage        = min(1.0, n_relevant_tickets / 2)
                          -- 2+ tickets for the affected customers, in/near
                          the anomaly week, earns full credit.
baseline_reliability   = min(1.0, n_baseline_weeks / 8)
                          -- pulled directly from detect.py's output.
customer_context       = (# affected customers with a customer_master row)
                          / (# affected customers total)

evidence_sufficiency   = min(ticket_coverage, baseline_reliability, customer_context)

attribution_clarity (revenue only):
  dominant_share = max(|volume|,|mix|,|price|) / (|volume|+|mix|+|price|)
  rescaled to 0-1 so that a perfectly 3-way split (dominant_share=1/3)
  maps to 0, and total dominance (dominant_share=1.0) maps to 1:
    attribution_clarity = (dominant_share - 1/3) / (1 - 1/3)

Confidence tiers (from evidence_sufficiency; thresholds documented, tunable):
  >= 0.7  -> "high"
  0.4-0.7 -> "medium"
  <  0.4  -> "low_confidence_abstain"

Run:
    python3 src/evidence_score.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

TICKET_DATE_WINDOW_DAYS = 7  # a ticket counts as "relevant" if within this many
                              # days of the anomaly week's start, for either
                              # of the affected customers.

HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.4


def get_affected_customers(transactions: pd.DataFrame, region: str, segment: str, week) -> list:
    week = pd.Timestamp(week)
    sub = transactions.copy()
    sub["order_date"] = pd.to_datetime(sub["order_date"])
    sub["week"] = sub["order_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    match = sub[(sub["region"] == region) & (sub["segment"] == segment) & (sub["week"] == week)]
    return sorted(match["customer_id"].unique().tolist())


def ticket_coverage_score(support_tickets: pd.DataFrame, customer_ids: list, week) -> tuple:
    week = pd.Timestamp(week)
    window_start = week - pd.Timedelta(days=TICKET_DATE_WINDOW_DAYS)
    window_end = week + pd.Timedelta(days=7 + TICKET_DATE_WINDOW_DAYS)  # week + 7 days buffer on both ends

    tix = support_tickets.copy()
    tix["ticket_date"] = pd.to_datetime(tix["ticket_date"], format="mixed")
    relevant = tix[
        tix["customer_id"].isin(customer_ids)
        & (tix["ticket_date"] >= window_start)
        & (tix["ticket_date"] <= window_end)
    ]
    score = min(1.0, len(relevant) / 2)
    return score, relevant


def baseline_reliability_score(n_baseline_weeks) -> float:
    if pd.isna(n_baseline_weeks):
        return 0.0
    return min(1.0, n_baseline_weeks / 8)


def customer_context_score(customer_master: pd.DataFrame, customer_ids: list) -> float:
    if not customer_ids:
        return 0.0
    present = customer_master["customer_id"].isin(customer_ids).sum()
    covered_ids = set(customer_master[customer_master["customer_id"].isin(customer_ids)]["customer_id"])
    return len(covered_ids) / len(customer_ids)


def attribution_clarity_score(volume_effect, mix_effect, price_effect):
    if pd.isna(volume_effect) or pd.isna(mix_effect) or pd.isna(price_effect):
        return None
    effects = [abs(volume_effect), abs(mix_effect), abs(price_effect)]
    total = sum(effects)
    if total == 0:
        return None
    dominant_share = max(effects) / total
    return (dominant_share - 1 / 3) / (1 - 1 / 3)


def confidence_tier(evidence_sufficiency: float) -> str:
    if evidence_sufficiency >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif evidence_sufficiency >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low_confidence_abstain"


def run_evidence_scoring(
    flagged_path: str = "data/processed/detect_results_flagged.csv",
    decompose_path: str = "data/processed/decompose_results.csv",
    transactions_path: str = "data/processed/transactions.csv",
    tickets_path: str = "data/processed/support_tickets.csv",
    customer_master_path: str = "data/processed/customer_master.csv",
) -> pd.DataFrame:
    flagged = pd.read_csv(flagged_path)
    decompose = pd.read_csv(decompose_path)
    transactions = pd.read_csv(transactions_path)
    tickets = pd.read_csv(tickets_path)
    customer_master = pd.read_csv(customer_master_path)

    merged = flagged.merge(
        decompose[["kpi", "region", "segment", "week", "volume_effect", "mix_effect", "price_effect"]],
        on=["kpi", "region", "segment", "week"], how="left",
    )

    results = []
    for _, row in merged.iterrows():
        region, segment, week = row["region"], row["segment"], row["week"]
        customer_ids = get_affected_customers(transactions, region, segment, week)

        tc_score, relevant_tickets = ticket_coverage_score(tickets, customer_ids, week)
        br_score = baseline_reliability_score(row["n_baseline_weeks"])
        cc_score = customer_context_score(customer_master, customer_ids)

        evidence_sufficiency = min(tc_score, br_score, cc_score)
        tier = confidence_tier(evidence_sufficiency)

        attribution = attribution_clarity_score(
            row.get("volume_effect"), row.get("mix_effect"), row.get("price_effect")
        )

        results.append({
            "kpi": row["kpi"], "region": region, "segment": segment, "week": week,
            "n_affected_customers": len(customer_ids),
            "n_relevant_tickets": len(relevant_tickets),
            "ticket_coverage_score": round(tc_score, 3),
            "baseline_reliability_score": round(br_score, 3),
            "customer_context_score": round(cc_score, 3),
            "evidence_sufficiency": round(evidence_sufficiency, 3),
            "confidence_tier": tier,
            "attribution_clarity": round(attribution, 3) if attribution is not None else None,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running evidence scoring on all flagged anomalies...\n")
    results = run_evidence_scoring()
    print(f"Scored {len(results)} flagged anomalies.\n")

    print("=== Validation: known East/Corporate revenue anomaly, week of 2017-12-11 ===")
    check = results[
        (results["kpi"] == "revenue") & (results["region"] == "East")
        & (results["segment"] == "Corporate") & (results["week"].astype(str) == "2017-12-11")
    ]
    pd.set_option("display.width", 200)
    print(check.to_string(index=False))

    if len(check):
        row = check.iloc[0]
        print(f"\nExpected: evidence_sufficiency=1.0, confidence_tier=high, attribution_clarity~0.36")
        print(f"Got: evidence_sufficiency={row['evidence_sufficiency']}, "
              f"confidence_tier={row['confidence_tier']}, attribution_clarity={row['attribution_clarity']}")

        checks_passed = (
            row["evidence_sufficiency"] == 1.0
            and row["confidence_tier"] == "high"
            and abs(row["attribution_clarity"] - 0.36) < 0.02
        )
        print("\nPASS: known scenario scores exactly as designed." if checks_passed
              else "\nFAIL: known scenario did not score as expected -- check the logic.")

    print("\n=== Distribution of confidence tiers across all flagged anomalies ===")
    print(results["confidence_tier"].value_counts())

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    results.to_csv("data/processed/evidence_scores.csv", index=False)
    print("\nWrote data/processed/evidence_scores.csv")
