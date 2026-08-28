"""
prepare_transactions.py

Purpose
-------
Takes the real Superstore dataset (data/raw/superstore_raw.csv) and produces
data/processed/transactions.csv, which is what the rest of the Lens pipeline
(reconcile.py, detect.py, decompose.py, ...) reads from.

Two things happen here:

1. Column standardization — real Superstore's column names ("Order Date",
   "Customer ID", etc.) are renamed to the snake_case schema used across
   contracts/*.yaml and the rest of the codebase (order_date, customer_id, ...).

2. ONE documented, minimal synthetic injection — see INJECTION LOG below.
   Everything else in the file (9,991 of 9,994 rows) is completely untouched
   real data.

Why an injection at all
------------------------
Round 2 requires a demonstrable "multi-factor KPI movement" scenario (price +
discount + volume moving together). We searched the real 2014-2017 Superstore
data for a naturally-occurring week that told this story cleanly and found:

  East / Corporate, week of 2017-12-11:
    - Order volume is ALREADY naturally low that week: 3 orders vs. an
      8-week trailing baseline average of ~5.5 orders/week. Real, unmodified.
    - Discount is ALREADY naturally pulled back that week: mean discount of
      0.067 vs. baseline mean of ~0.165. Real, unmodified.
    - Even before any changes, this week already crosses our detection
      thresholds (modified z-score -1.86, revenue -80.6% vs. baseline
      median) using ONLY real, unmodified data.

  The only missing element to complete a genuine "price + discount + volume"
  three-lever story was a price increase. Rather than fabricate new rows,
  we applied a single documented multiplier to the Sales (and proportionally
  Profit) values of the 3 REAL order lines that already exist in that week,
  for that region/segment. Customer IDs, order IDs, dates, categories, and
  all other weeks/regions/segments are 100% real and untouched.

INJECTION LOG (also written to docs/injection_log.md at runtime)
------------------------------------------------------------------
  Scope: Region == 'East', Segment == 'Corporate', week of 2017-12-11
  Rows affected: 3 (out of 9,994 total rows in the dataset)
  Change applied: Sales *= 1.4, Profit *= 1.4 (preserves each row's original
                  profit margin ratio)
  Rationale: introduces a "price increase" lever on top of the two levers
             (volume drop, discount pull-back) that were already real and
             naturally occurring in that week, producing a defensible
             multi-factor revenue-decline scenario for the Detect/Decompose/
             Explain pipeline to reason about.

Run:
    python3 scripts/prepare_transactions.py
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/superstore_raw.csv")
OUT_PATH = Path("data/processed/transactions.csv")
LOG_PATH = Path("docs/injection_log.md")

# --- Injection parameters (see docstring above for rationale) ---
INJECT_REGION = "East"
INJECT_SEGMENT = "Corporate"
INJECT_WEEK_START = pd.Timestamp("2017-12-11")  # W-SUN week bucket
PRICE_MULTIPLIER = 1.4


def load_and_standardize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")

    df = df.rename(columns={
        "Customer ID": "customer_id",
        "Order ID": "order_id",
        "Order Date": "order_date",
        "Ship Date": "ship_date",
        "Region": "region",
        "Segment": "segment",
        "Category": "category",
        "Sub-Category": "sub_category",
        "Product Name": "product_name",
        "Sales": "sales",
        "Quantity": "quantity",
        "Discount": "discount",
        "Profit": "profit",
        "State": "state",
        "City": "city",
    })

    keep_cols = [
        "customer_id", "order_id", "order_date", "ship_date", "region",
        "segment", "category", "sub_category", "product_name", "sales",
        "quantity", "discount", "profit", "state", "city",
    ]
    return df[keep_cols]


def apply_injection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies the single documented injection. Returns (modified_df, log_df)
    where log_df captures the exact before/after values of every changed row."""
    df = df.copy()
    df["week"] = df["order_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    mask = (
        (df["region"] == INJECT_REGION)
        & (df["segment"] == INJECT_SEGMENT)
        & (df["week"] == INJECT_WEEK_START)
    )
    affected = df[mask].copy()
    if len(affected) == 0:
        raise RuntimeError(
            "Injection target not found — did the raw data change? "
            f"Expected rows for region={INJECT_REGION}, segment={INJECT_SEGMENT}, "
            f"week={INJECT_WEEK_START.date()}."
        )

    log_rows = []
    for idx in affected.index:
        before_sales = df.loc[idx, "sales"]
        before_profit = df.loc[idx, "profit"]
        after_sales = round(before_sales * PRICE_MULTIPLIER, 4)
        after_profit = round(before_profit * PRICE_MULTIPLIER, 4)

        log_rows.append({
            "order_id": df.loc[idx, "order_id"],
            "customer_id": df.loc[idx, "customer_id"],
            "order_date": df.loc[idx, "order_date"].date(),
            "category": df.loc[idx, "category"],
            "sales_before": before_sales,
            "sales_after": after_sales,
            "profit_before": before_profit,
            "profit_after": after_profit,
            "multiplier": PRICE_MULTIPLIER,
        })

        df.loc[idx, "sales"] = after_sales
        df.loc[idx, "profit"] = after_profit

    df = df.drop(columns=["week"])
    return df, pd.DataFrame(log_rows)


def write_injection_log(log_df: pd.DataFrame, path: Path):
    lines = [
        "# Injection Log — Lens Round 2\n",
        "This file documents the ONLY synthetic modification made to the real",
        "Superstore dataset used in this prototype. All other 9,991 rows are",
        "100% real, unmodified Kaggle Superstore data.\n",
        f"**Scope:** region=`{INJECT_REGION}`, segment=`{INJECT_SEGMENT}`, "
        f"week starting `{INJECT_WEEK_START.date()}`\n",
        f"**Change:** `sales` and `profit` multiplied by **{PRICE_MULTIPLIER}** "
        "on the rows below (margin ratio preserved).\n",
        "**Why:** this week already had a real, unmodified volume drop "
        "(3 orders vs. ~5.5 baseline) and a real, unmodified discount "
        "pull-back (0.067 vs. ~0.165 baseline mean). This injection adds "
        "the third lever — a price increase — to complete an honest "
        "multi-factor revenue-decline scenario for the Detect/Decompose/"
        "Explain pipeline.\n",
        "## Rows modified\n",
        log_df.to_markdown(index=False),
        "\n",
    ]
    path.write_text("\n".join(lines))


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_and_standardize(RAW_PATH)
    print(f"Loaded {len(df)} real rows.")

    print("Applying documented injection ...")
    df, log_df = apply_injection(df)
    print(f"Modified {len(log_df)} rows (out of {len(df)} total).")
    print(log_df.to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote standardized + injected data to {OUT_PATH}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_injection_log(log_df, LOG_PATH)
    print(f"Wrote injection log to {LOG_PATH}")


if __name__ == "__main__":
    main()
