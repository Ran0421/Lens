"""
decompose.py — Stage 3 of the Lens pipeline: Decompose

Non-LLM. Pure deterministic arithmetic. Takes every flagged anomaly from
Detect and explains WHICH slice/driver contributed to it, using two
complementary methods:

1. Category contribution breakdown — for any flagged KPI, shows how each
   product category's value changed vs. baseline, ranked by contribution.

2. Volume / Mix / Price waterfall (revenue only) — the standard FP&A
   technique for explaining a revenue change in terms of:
     - Volume effect:  did total order count change?
     - Mix effect:     did orders shift toward higher/lower-priced categories?
     - Price effect:   did prices within a category actually change?

A note on why this is 3 factors, not 2
-----------------------------------------
A naive 2-factor split (volume x price, computed only at the aggregate
level) has NO room for a separate mix term -- volume_effect + price_effect
reconciles EXACTLY to total_change with nothing left over. That's an
algebraic identity, not a finding. A genuine mix effect only exists once
you decompose at the category level: it isolates whether the ORDER MIX
shifted toward different categories, separately from whether any category's
price actually moved. This file implements the category-level 3-way
version so "mix" means something real.

The math (per flagged revenue anomaly)
-----------------------------------------
Let baseline_price_c = baseline avg order value in category c
    baseline_overall_price = blended baseline avg order value (all categories)
    current_orders_c, current_price_c = same, for the flagged week

  volume_effect = (current_total_orders - baseline_total_orders) * baseline_overall_price
  mix_effect    = sum_c(current_orders_c * baseline_price_c) - current_total_orders * baseline_overall_price
  price_effect  = sum_c(current_orders_c * (current_price_c - baseline_price_c))

These three sum EXACTLY to total_change (current_revenue - expected baseline
revenue). A reconciliation_error field is included in the output specifically
so this identity can be checked, not assumed.

Non-revenue KPIs (order_volume, discount_rate, profit_margin)
-----------------------------------------------------------------
These aren't cleanly additive across categories the way revenue is (you
can't "sum" a discount rate or a margin meaningfully). For these, we only
compute the category contribution breakdown -- which category's activity
shifted the most vs. its own baseline -- not the volume/mix/price waterfall.

Run:
    python3 src/decompose.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

TRAILING_WINDOW_WEEKS = 8  # must match detect.py


def _get_slice_weeks(df: pd.DataFrame, region: str, segment: str) -> pd.DataFrame:
    sub = df[(df["region"] == region) & (df["segment"] == segment)].copy()
    sub["order_date"] = pd.to_datetime(sub["order_date"])
    sub["week"] = sub["order_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    return sub


def _category_breakdown(current_df: pd.DataFrame, baseline_df: pd.DataFrame, kpi: str) -> list:
    """Generic category-level contribution breakdown for any KPI."""
    breakdown = []
    all_categories = set(current_df["category"]).union(set(baseline_df["category"]))
    n_baseline_weeks = baseline_df["week"].nunique() if len(baseline_df) else 0

    for cat in all_categories:
        cur_cat = current_df[current_df["category"] == cat]
        base_cat = baseline_df[baseline_df["category"] == cat]

        if kpi == "revenue":
            cur_val = cur_cat["sales"].sum()
            base_val = base_cat["sales"].sum() / n_baseline_weeks if n_baseline_weeks else 0.0
        elif kpi == "order_volume":
            cur_val = cur_cat["order_id"].nunique()
            base_val = base_cat["order_id"].nunique() / n_baseline_weeks if n_baseline_weeks else 0.0
        elif kpi == "discount_rate":
            cur_val = cur_cat["discount"].mean() if len(cur_cat) else 0.0
            base_val = base_cat["discount"].mean() if len(base_cat) else 0.0
        elif kpi == "profit_margin":
            cur_val = (cur_cat["profit"].sum() / cur_cat["sales"].sum()) if cur_cat["sales"].sum() else 0.0
            base_val = (base_cat["profit"].sum() / base_cat["sales"].sum()) if base_cat["sales"].sum() else 0.0
        else:
            continue

        delta = cur_val - base_val
        breakdown.append({
            "category": cat,
            "current_value": round(float(cur_val), 4),
            "baseline_value": round(float(base_val), 4),
            "delta": round(float(delta), 4),
        })

    breakdown.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return breakdown


def _volume_mix_price(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """The 3-way waterfall, revenue only. See module docstring for the math."""
    categories = set(current_df["category"]).union(set(baseline_df["category"]))
    n_baseline_weeks = baseline_df["week"].nunique()

    baseline_total_orders = baseline_df["order_id"].nunique() / n_baseline_weeks if n_baseline_weeks else 0.0
    baseline_total_revenue = baseline_df["sales"].sum() / n_baseline_weeks if n_baseline_weeks else 0.0
    baseline_overall_price = (baseline_total_revenue / baseline_total_orders) if baseline_total_orders else 0.0

    current_total_orders = current_df["order_id"].nunique()
    current_total_revenue = current_df["sales"].sum()

    volume_effect = (current_total_orders - baseline_total_orders) * baseline_overall_price

    mix_term_sum = 0.0
    price_term_sum = 0.0
    for cat in categories:
        base_cat = baseline_df[baseline_df["category"] == cat]
        cur_cat = current_df[current_df["category"] == cat]

        base_cat_orders = (base_cat["order_id"].nunique() / n_baseline_weeks) if n_baseline_weeks else 0.0
        base_cat_revenue = (base_cat["sales"].sum() / n_baseline_weeks) if n_baseline_weeks else 0.0
        # Fallback to blended baseline price if this category has no baseline history
        # (e.g. a brand-new category in this slice) -- avoids divide-by-zero and
        # avoids a spurious price-effect signal for a category we have no baseline for.
        base_cat_price = (base_cat_revenue / base_cat_orders) if base_cat_orders else baseline_overall_price

        cur_cat_orders = cur_cat["order_id"].nunique()
        cur_cat_revenue = cur_cat["sales"].sum()
        cur_cat_price = (cur_cat_revenue / cur_cat_orders) if cur_cat_orders else 0.0

        mix_term_sum += cur_cat_orders * base_cat_price
        price_term_sum += cur_cat_orders * (cur_cat_price - base_cat_price)

    mix_effect = mix_term_sum - current_total_orders * baseline_overall_price
    price_effect = price_term_sum

    total_change = current_total_revenue - baseline_total_revenue
    reconciliation_error = total_change - (volume_effect + mix_effect + price_effect)

    effects = {"volume_effect": volume_effect, "mix_effect": mix_effect, "price_effect": price_effect}
    top_driver = max(effects, key=lambda k: abs(effects[k]))

    return {
        "baseline_avg_weekly_orders": round(baseline_total_orders, 2),
        "baseline_avg_weekly_revenue": round(baseline_total_revenue, 2),
        "baseline_avg_price": round(baseline_overall_price, 2),
        "current_orders": current_total_orders,
        "current_revenue": round(current_total_revenue, 2),
        "total_change": round(total_change, 2),
        "volume_effect": round(volume_effect, 2),
        "mix_effect": round(mix_effect, 2),
        "price_effect": round(price_effect, 2),
        "reconciliation_error": round(reconciliation_error, 4),  # should be ~0
        "top_driver": top_driver,
    }


def validate_against_known_injection(
    transactions_path: str = "data/processed/transactions.csv",
) -> dict:
    """Isolates the effect of the documented injection (see
    scripts/prepare_transactions.py / docs/injection_log.md) by re-running
    the volume/mix/price waterfall with the 3 injected rows reverted to
    their real, pre-injection values, and comparing to the actual pipeline
    output.

    This is a permanent, reproducible correctness check, not a one-off:
    it confirms that decompose.py attributes the injection's effect to
    price_effect specifically (as it should, since the injection only
    changed price, not volume or category mix) and to nothing else.
    """
    INJECTED_ORDER_IDS = ["CA-2017-129581", "CA-2017-135377", "US-2017-125213"]
    INJECTION_MULTIPLIER = 1.4

    txns = pd.read_csv(transactions_path)
    txns["order_date"] = pd.to_datetime(txns["order_date"])

    slice_df = _get_slice_weeks(txns, "East", "Corporate")
    week = pd.Timestamp("2017-12-11")
    current_with_injection = slice_df[slice_df["week"] == week]

    all_weeks = sorted(slice_df["week"].unique())
    prior_weeks = [w for w in all_weeks if w < week][-TRAILING_WINDOW_WEEKS:]
    baseline_df = slice_df[slice_df["week"].isin(prior_weeks)]

    current_without_injection = current_with_injection.copy()
    mask = current_without_injection["order_id"].isin(INJECTED_ORDER_IDS)
    current_without_injection.loc[mask, "sales"] /= INJECTION_MULTIPLIER
    current_without_injection.loc[mask, "profit"] /= INJECTION_MULTIPLIER

    with_result = _volume_mix_price(current_with_injection, baseline_df)
    without_result = _volume_mix_price(current_without_injection, baseline_df)

    revenue_delta = with_result["current_revenue"] - without_result["current_revenue"]
    price_effect_delta = with_result["price_effect"] - without_result["price_effect"]
    # The two deltas should match closely -- the injection only touched price,
    # so its entire contribution should flow through price_effect and nowhere else.
    isolation_error = abs(revenue_delta - price_effect_delta)

    return {
        "with_injection": with_result,
        "without_injection": without_result,
        "revenue_delta_from_injection": round(revenue_delta, 2),
        "price_effect_delta_from_injection": round(price_effect_delta, 2),
        "isolation_error": round(isolation_error, 4),  # should be ~0
    }


def run_decompose(
    flagged_path: str = "data/processed/detect_results_flagged.csv",
    transactions_path: str = "data/processed/transactions.csv",
) -> pd.DataFrame:
    flagged = pd.read_csv(flagged_path)
    txns = pd.read_csv(transactions_path)
    txns["order_date"] = pd.to_datetime(txns["order_date"])

    results = []
    for _, anomaly in flagged.iterrows():
        region, segment, kpi = anomaly["region"], anomaly["segment"], anomaly["kpi"]
        week = pd.Timestamp(anomaly["week"])

        slice_df = _get_slice_weeks(txns, region, segment)
        current_df = slice_df[slice_df["week"] == week]

        all_weeks_sorted = sorted(slice_df["week"].unique())
        prior_weeks = [w for w in all_weeks_sorted if w < week][-TRAILING_WINDOW_WEEKS:]
        baseline_df = slice_df[slice_df["week"].isin(prior_weeks)]

        category_breakdown = _category_breakdown(current_df, baseline_df, kpi)
        top_category = category_breakdown[0]["category"] if category_breakdown else None

        record = {
            "kpi": kpi, "region": region, "segment": segment, "week": week.date(),
            "detect_pct_change": anomaly["pct_change"],
            "top_category": top_category,
            "category_breakdown": json.dumps(category_breakdown),
        }

        if kpi == "revenue":
            waterfall = _volume_mix_price(current_df, baseline_df)
            record.update(waterfall)
        else:
            record.update({
                "baseline_avg_weekly_orders": None, "baseline_avg_weekly_revenue": None,
                "baseline_avg_price": None, "current_orders": None, "current_revenue": None,
                "total_change": None, "volume_effect": None, "mix_effect": None,
                "price_effect": None, "reconciliation_error": None, "top_driver": "category_shift",
            })

        results.append(record)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running Decompose on all flagged anomalies...\n")
    results = run_decompose()
    print(f"Decomposed {len(results)} flagged anomalies.\n")

    print("=== Validation: known East/Corporate revenue anomaly, week of 2017-12-11 ===")
    check = results[
        (results["kpi"] == "revenue") & (results["region"] == "East")
        & (results["segment"] == "Corporate") & (results["week"].astype(str) == "2017-12-11")
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)
    cols = ["kpi", "region", "segment", "week", "total_change", "volume_effect",
            "mix_effect", "price_effect", "reconciliation_error", "top_driver"]
    print(check[cols].to_string(index=False))

    if len(check):
        row = check.iloc[0]
        print(f"\nReconciliation check (should be ~0): {row['reconciliation_error']}")
        print(f"Volume effect: {row['volume_effect']} (expect large negative -- volume collapsed)")
        print(f"Price effect: {row['price_effect']} (net negative overall -- see isolation check below)")
        print(f"Top driver: {row['top_driver']}")
        print("\nCategory breakdown for this anomaly:")
        print(json.dumps(json.loads(row["category_breakdown"]), indent=2))

    print("\n=== Injection isolation check ===")
    print("Confirms the documented price injection's effect is correctly isolated")
    print("to price_effect specifically, and nowhere else in the decomposition.\n")
    iso = validate_against_known_injection()
    print(f"Revenue WITH injection:    {iso['with_injection']['current_revenue']}")
    print(f"Revenue WITHOUT injection: {iso['without_injection']['current_revenue']}")
    print(f"Revenue delta from injection:      {iso['revenue_delta_from_injection']}")
    print(f"price_effect delta from injection: {iso['price_effect_delta_from_injection']}")
    print(f"Isolation error (should be ~0):    {iso['isolation_error']}")
    if iso["isolation_error"] < 0.05:  # tolerance for rounding noise from 2dp-rounded intermediate fields
        print("PASS: injection's effect is fully and correctly isolated to price_effect.")
    else:
        print("FAIL: injection's effect leaked into volume_effect or mix_effect -- check the math.")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    results.to_csv("data/processed/decompose_results.csv", index=False)
    print("\nWrote data/processed/decompose_results.csv")
