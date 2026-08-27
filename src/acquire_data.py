"""
Acquire and construct the three data sources for Lens Round 2.

IMPORTANT DESIGN NOTE (read before modifying):
Superstore, Telco Churn, and the Customer Support Ticket dataset are three
UNRELATED real companies -- they do not share customers, and joining them
directly on customer_id would be a fabricated relationship, not a real one.

Instead: Superstore is the single source of truth for identity (its
customer_id, order_id, and region are real and internally consistent).
The other two sources are CONSTRUCTED from Superstore's real customer list,
using Telco and the ticket dataset only as calibration/style donors:

  - Source B (customer master): every row's customer_id is a real Superstore
    customer_id. tenure_months and churn_flag are sampled from Telco's real
    *distribution* (so the numbers behave like a genuine customer base),
    not copied from Telco's (unrelated) customers directly.
  - Source C (support tickets): every row's customer_id and ticket_date are
    remapped to a real Superstore customer/order window. The ticket TEXT
    itself is sampled verbatim from the real ticket dataset (so the
    language is authentic), only the attribution is reassigned.

This means every SQL join in reconcile.py is a real primary-key
relationship on customer_id, not a coincidental one -- while every number
and sentence in the derived tables still traces back to real data.
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_sources():
    """Download all three raw datasets via kagglehub. Run once."""
    import kagglehub

    paths = {}
    paths["superstore"] = kagglehub.dataset_download("vivek468/superstore-dataset-final")
    paths["telco"] = kagglehub.dataset_download("blastchar/telco-customer-churn")
    paths["tickets"] = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
    return paths


# ---------------------------------------------------------------------------
# Source A: Transactions (daily grain) -- real, internally consistent as-is
# ---------------------------------------------------------------------------
def load_transactions(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, encoding="latin1")
    df["order_date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["order_date"])
    df = df.rename(columns={
        "Region": "region",
        "Segment": "segment",
        "Category": "category",
        "Sub-Category": "sub_category",
        "Sales": "sales",
        "Profit": "profit",
        "Discount": "discount",
        "Order ID": "order_id",
        "Customer ID": "customer_id",
    })
    keep = ["order_date", "region", "segment", "category", "sub_category",
            "sales", "profit", "discount", "order_id", "customer_id"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["source"] = "transactions"
    df["grain"] = "daily"
    return df


# ---------------------------------------------------------------------------
# Source B: Customer master (weekly-snapshot grain)
# Real customer_id (from Superstore) + calibrated-synthetic attributes
# (distribution borrowed from Telco, not copied from Telco's own customers)
# ---------------------------------------------------------------------------
def build_customer_master(transactions: pd.DataFrame, telco_raw: pd.DataFrame,
                           as_of_week: str, seed: int = 42) -> pd.DataFrame:
    """
    For every real Superstore customer_id, construct a weekly snapshot row.
    tenure_months, contract_type, and churn_flag are drawn from Telco's
    empirical distribution via sampling-with-replacement -- this preserves
    realistic shape (e.g. Telco's actual tenure skew, actual churn rate)
    without claiming any individual Telco customer IS a Superstore customer.
    """
    rng = np.random.default_rng(seed=seed)

    real_customers = (
        transactions[["customer_id", "region"]]
        .drop_duplicates(subset="customer_id")
        .reset_index(drop=True)
    )

    telco_clean = telco_raw.rename(columns={
        "tenure": "tenure_months",
        "Contract": "contract_type",
        "MonthlyCharges": "monthly_charges",
        "Churn": "churn_flag",
    })
    telco_clean["churn_flag"] = (telco_clean["churn_flag"] == "Yes").astype(int)

    sample_idx = rng.integers(0, len(telco_clean), size=len(real_customers))
    sampled_attrs = telco_clean.iloc[sample_idx][
        ["tenure_months", "contract_type", "monthly_charges", "churn_flag"]
    ].reset_index(drop=True)

    out = pd.concat([real_customers, sampled_attrs], axis=1)
    out["snapshot_week"] = pd.to_datetime(as_of_week)
    out["source"] = "customer_master"
    out["grain"] = "weekly"
    out["attribute_provenance"] = "sampled from Telco distribution, attached to real Superstore customer_id"
    return out


# ---------------------------------------------------------------------------
# Source C: Support tickets (event-level, unstructured)
# Real ticket TEXT, remapped to real Superstore customer_id + order window
# ---------------------------------------------------------------------------
def build_support_tickets(transactions: pd.DataFrame, tickets_raw: pd.DataFrame,
                           tickets_per_customer_range=(0, 3), seed: int = 42) -> pd.DataFrame:
    """
    Sample real ticket text (and category, if present) from the ticket
    dataset, then attach each sampled ticket to a real Superstore
    customer_id and a real order_date drawn from that customer's own order
    history -- so tickets land inside a plausible window relative to actual
    purchase activity, rather than at an arbitrary unrelated date.
    """
    rng = np.random.default_rng(seed=seed)

    rename_map = {c: c.strip().lower().replace(" ", "_") for c in tickets_raw.columns}
    tickets_raw = tickets_raw.rename(columns=rename_map)

    text_col = next((c for c in tickets_raw.columns
                      if "description" in c or "text" in c or "body" in c), tickets_raw.columns[0])

    customer_orders = transactions[["customer_id", "region", "segment", "order_date"]]
    real_customers = customer_orders["customer_id"].unique()

    rows = []
    for cust in real_customers:
        n_tickets = rng.integers(*tickets_per_customer_range)
        if n_tickets == 0:
            continue
        cust_orders = customer_orders[customer_orders["customer_id"] == cust]
        sampled_tickets = tickets_raw.sample(n=n_tickets, random_state=int(rng.integers(0, 1_000_000)))
        sampled_dates = cust_orders.sample(n=n_tickets, replace=True,
                                            random_state=int(rng.integers(0, 1_000_000)))
        for (_, ticket_row), (_, order_row) in zip(sampled_tickets.iterrows(), sampled_dates.iterrows()):
            rows.append({
                "customer_id": cust,
                "region": order_row["region"],
                "segment": order_row["segment"],
                "ticket_date": order_row["order_date"],
                "ticket_text": ticket_row[text_col],
            })

    out = pd.DataFrame(rows)
    out["source"] = "support_tickets"
    out["grain"] = "event"
    out["attribute_provenance"] = "real ticket text, remapped to real Superstore customer_id + order window"
    return out


def synthesize_scenario_tickets(scenario_evidence: list[dict]) -> pd.DataFrame:
    """
    Overlay a handful of scenario-specific synthetic tickets on top of the
    constructed ticket corpus above -- used only to make the 4 required
    demo scenarios concretely walkable (e.g. tickets that mention pricing
    in the exact week/region a Detect-stage anomaly fires). Clearly
    disclosed as fully synthetic, unlike build_support_tickets() above
    which reuses real ticket text. See docs/architecture.md.

    scenario_evidence: list of dicts with keys
        {ticket_date, region, segment, ticket_text}
    """
    df = pd.DataFrame(scenario_evidence)
    df["source"] = "support_tickets_scenario_synthetic"
    df["grain"] = "event"
    df["attribute_provenance"] = "fully synthetic, written for demo scenario"
    return df


if __name__ == "__main__":
    print("This module is imported by src/reconcile.py.")
    print("Run download_sources() once with network access to populate data/raw/.")
