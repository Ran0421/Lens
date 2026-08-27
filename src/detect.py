"""
Detect: flags genuine KPI anomalies against a rolling seasonal baseline.

Non-LLM stage. Pure statistics -- deliberately so. This is the first check
in the pipeline and everything downstream (Decompose, Explain, Recommend)
trusts its output, so it must be fully deterministic and explainable
without an LLM in the loop.

Materiality rule (documented decision, not a silent default):
A slice is flagged if EITHER the statistical test OR the business test
trips (combine_rule: "OR" in every KPI contract). This is the more
sensitive option -- it catches more real movements at the cost of more
alerts. The stricter alternative (AND: both tests must trip) would miss
a high-dollar-impact move that isn't statistically wild in a naturally
noisy slice. OR was chosen because the brief's own real-world complexity
list explicitly calls under-flagging "real liability", and materiality is
meant to be *tuned*, not solved away -- see each contract's
`combine_rule` field, which is what actually governs this at runtime, not
this docstring. Changing the tradeoff means editing the YAML contract, not
this code.
"""

import pandas as pd
import numpy as np
import yaml
import os


def load_contract(kpi_name: str, contracts_dir: str = "contracts") -> dict:
    path = os.path.join(contracts_dir, f"{kpi_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _rolling_baseline(series: pd.Series, window_periods: int) -> pd.DataFrame:
    """
    Trailing rolling mean/std, EXCLUDING the current period from its own
    baseline (shift(1)) -- otherwise an anomaly would pull its own
    baseline toward itself and partially mask its own detection.
    """
    shifted = series.shift(1)
    mean = shifted.rolling(window=window_periods, min_periods=3).mean()
    std = shifted.rolling(window=window_periods, min_periods=3).std()
    return pd.DataFrame({"baseline_mean": mean, "baseline_std": std})


def detect_anomalies(daily_agg: pd.DataFrame, kpi_name: str,
                      contracts_dir: str = "contracts",
                      threshold_overrides: dict = None) -> pd.DataFrame:
    """
    daily_agg: the reconcile.Warehouse `daily_agg` table (or
               `weekly_repeat_purchase` for that KPI), must contain a date
               column, `region`, `segment` (if present), and the KPI's
               value_column as specified in its contract.

    threshold_overrides: OPTIONAL dict of materiality_threshold fields to
               override for THIS CALL ONLY -- the underlying YAML contract
               is never modified. Exists for exactly one legitimate reason:
               percentage-based business thresholds calibrated for a real,
               high-volume dataset become unreliable noise at small sample
               sizes (a slice with 1-3 orders/day naturally swings >25% day
               to day with no anomaly present at all -- verified empirically
               against data/sample/, see docs/detect_threshold_tuning.md).
               The statistical (z-score) test does NOT have this problem,
               since it's variance-aware by construction -- only the
               business_pct_threshold needs adjusting at small scale.
               Production runs against the full downloaded datasets should
               NOT pass overrides; the contract's real thresholds apply.

    Returns one row per (date, region[, segment]) with baseline stats,
    both test results, and the combined anomaly flag + which test(s)
    tripped -- so every flag is traceable to a specific number, not an
    opaque boolean.
    """
    contract = load_contract(kpi_name, contracts_dir)
    thresh = dict(contract["materiality_threshold"])
    if threshold_overrides:
        thresh.update(threshold_overrides)

    value_col = thresh["value_column"]
    std_mult = thresh["std_multiplier"]
    window = thresh["window_periods"]
    abs_thresh = thresh.get("business_abs_threshold")
    pct_thresh = thresh.get("business_pct_threshold")
    combine_rule = thresh.get("combine_rule", "OR")

    date_col = "date" if "date" in daily_agg.columns else "week_ending"
    group_cols = [c for c in ["region", "segment"] if c in daily_agg.columns]

    df = daily_agg.sort_values(date_col).copy()

    results = []
    for keys, group in df.groupby(group_cols):
        group = group.sort_values(date_col).reset_index(drop=True)
        baseline = _rolling_baseline(group[value_col], window)
        group = pd.concat([group, baseline], axis=1)

        group["z_score"] = (group[value_col] - group["baseline_mean"]) / group["baseline_std"]
        group["delta_abs"] = group[value_col] - group["baseline_mean"]
        group["delta_pct"] = group["delta_abs"] / group["baseline_mean"].replace(0, np.nan)

        stat_trip = group["z_score"].abs() > std_mult
        biz_trip = pd.Series(False, index=group.index)
        if abs_thresh is not None:
            biz_trip = biz_trip | (group["delta_abs"].abs() > abs_thresh)
        if pct_thresh is not None:
            biz_trip = biz_trip | (group["delta_pct"].abs() > pct_thresh)

        group["statistical_test_tripped"] = stat_trip.fillna(False)
        group["business_test_tripped"] = biz_trip.fillna(False)

        if combine_rule == "OR":
            group["is_anomaly"] = group["statistical_test_tripped"] | group["business_test_tripped"]
        else:  # "AND"
            group["is_anomaly"] = group["statistical_test_tripped"] & group["business_test_tripped"]

        def _reason(row):
            reasons = []
            if row["statistical_test_tripped"]:
                reasons.append(f"statistical (|z|={row['z_score']:.2f} > {std_mult})")
            if row["business_test_tripped"]:
                reasons.append("business (impact threshold exceeded)")
            return " AND ".join(reasons) if reasons else "none"

        group["materiality_reason"] = group.apply(_reason, axis=1)

        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                group[col] = val
        else:
            group[group_cols[0]] = keys

        results.append(group)

    out = pd.concat(results, ignore_index=True)
    out["kpi"] = kpi_name
    return out


def flagged_only(detect_result: pd.DataFrame) -> pd.DataFrame:
    """Convenience filter -- only the rows that actually tripped a test."""
    return detect_result[detect_result["is_anomaly"]].copy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.reconcile import Warehouse
    import pandas as pd

    transactions = pd.read_csv("data/sample/transactions_sample.csv", parse_dates=["order_date"])
    customer_master = pd.read_csv("data/sample/customer_master_sample.csv", parse_dates=["snapshot_week"])
    tickets = pd.read_csv("data/sample/support_tickets_sample.csv", parse_dates=["ticket_date"])

    wh = Warehouse()
    wh.load(transactions, customer_master, tickets)
    daily_agg = wh.query("SELECT * FROM daily_agg")

    # SAMPLE-SCALE OVERRIDE (dev/test only -- see docstring above).
    # business_pct_threshold=None disables the noisy percentage test at
    # small sample size; business_abs_threshold=700 is empirically tuned
    # against data/sample/'s actual noise distribution (median day-to-day
    # swing ~$420, the real injected anomaly's delta is ~$900-1200) --
    # see docs/detect_threshold_tuning.md for the full derivation.
    SAMPLE_OVERRIDE = {"business_pct_threshold": None, "business_abs_threshold": 700}

    result = detect_anomalies(daily_agg, "revenue", threshold_overrides=SAMPLE_OVERRIDE)
    flagged = flagged_only(result)
    print(f"Total slice-days evaluated: {len(result)}")
    print(f"Flagged anomalies: {len(flagged)} ({len(flagged)/len(result)*100:.0f}%)")
    print(flagged[["date", "region", "segment", "revenue", "baseline_mean",
                    "z_score", "materiality_reason"]].sort_values("date").to_string(index=False))
