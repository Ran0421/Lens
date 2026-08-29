"""
build_evidence_sources.py

Runs acquire_data.py's build_customer_master() and build_support_tickets()
against the REAL 793-customer Superstore dataset (data/processed/transactions.csv)
-- replacing the old sample files that were built against the 15-customer
synthetic sample, which is now retired.

Also overlays a small number of clearly-labeled SYNTHETIC scenario tickets
(via synthesize_scenario_tickets()) for the 3 real customers in our
documented demo anomaly (East/Corporate, week of 2017-12-11), so Explain
has something concrete to retrieve and cite for that scenario. These are
disclosed via source="support_tickets_scenario_synthetic" and are a small
minority of the overall ticket corpus -- the great majority of tickets in
support_tickets.csv are real ticket text remapped to real customers.

Run:
    python3 scripts/build_evidence_sources.py
"""

import sys
sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
from acquire_data import build_customer_master, build_support_tickets, synthesize_scenario_tickets

TRANSACTIONS_PATH = "data/processed/transactions.csv"
TELCO_RAW_PATH = "data/raw/telco_raw.csv"
TICKETS_RAW_PATH = "data/raw/tickets_raw.csv"

AS_OF_WEEK = "2017-12-25"  # last real week in the dataset -- customer_master is
                           # a current snapshot, not a per-week history (see
                           # design discussion: repeat-purchase-rate is computed
                           # directly from transactions in reconcile.py, so
                           # customer_master's job is just supplying contextual
                           # attributes for Explain to cite).

# The 3 real orders in our documented demo anomaly (East/Corporate,
# week of 2017-12-11 -- see docs/injection_log.md). Synthetic scenario
# tickets are attached to these customers, dated on their real order dates,
# so Explain has concrete evidence to retrieve for this specific scenario.
DEMO_SCENARIO_TICKETS = [
    {
        "customer_id": "KN-16390",
        "region": "East",
        "segment": "Corporate",
        "ticket_date": "2017-12-11",
        "ticket_text": (
            "Noticed the price on this Technology order is higher than what "
            "I was quoted last month. Was there a price change I wasn't told about?"
        ),
    },
    {
        "customer_id": "BP-11095",
        "region": "East",
        "segment": "Corporate",
        "ticket_date": "2017-12-13",
        "ticket_text": (
            "We usually get a corporate discount on Furniture orders this size "
            "but didn't see one applied this time. Can you confirm if that "
            "program is still active?"
        ),
    },
    {
        "customer_id": "NB-18655",
        "region": "East",
        "segment": "Corporate",
        "ticket_date": "2017-12-11",
        "ticket_text": (
            "Placed a much smaller order than usual this month -- just holding "
            "off on the bigger Office Supplies restock until pricing settles down."
        ),
    },
]


def main():
    print(f"Loading real transactions from {TRANSACTIONS_PATH} ...")
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    n_customers = transactions["customer_id"].nunique()
    print(f"Loaded {len(transactions)} rows, {n_customers} unique real customers.\n")

    print(f"Loading Telco raw ({TELCO_RAW_PATH}) and tickets raw ({TICKETS_RAW_PATH}) ...")
    telco_raw = pd.read_csv(TELCO_RAW_PATH)
    tickets_raw = pd.read_csv(TICKETS_RAW_PATH)
    print(f"Telco: {len(telco_raw)} rows. Tickets: {len(tickets_raw)} rows.\n")

    print(f"Building customer_master (single snapshot as of {AS_OF_WEEK}) ...")
    customer_master = build_customer_master(transactions, telco_raw, as_of_week=AS_OF_WEEK)
    print(f"Built {len(customer_master)} customer_master rows "
          f"(should equal {n_customers} unique customers).\n")

    print("Building support_tickets (real ticket text, remapped to real customers) ...")
    support_tickets_real = build_support_tickets(
        transactions, tickets_raw, tickets_per_customer_range=(0, 3), seed=42
    )
    print(f"Built {len(support_tickets_real)} real remapped ticket rows.\n")

    print("Overlaying documented synthetic scenario tickets for the demo anomaly ...")
    scenario_tickets = synthesize_scenario_tickets(DEMO_SCENARIO_TICKETS)
    print(f"Added {len(scenario_tickets)} clearly-labeled synthetic tickets "
          f"for customers {[t['customer_id'] for t in DEMO_SCENARIO_TICKETS]}.\n")

    support_tickets = pd.concat([support_tickets_real, scenario_tickets], ignore_index=True)
    print(f"Combined support_tickets: {len(support_tickets)} rows "
          f"({len(support_tickets_real)} real + {len(scenario_tickets)} synthetic, "
          f"{len(scenario_tickets)/len(support_tickets)*100:.2f}% synthetic).\n")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    customer_master.to_csv("data/processed/customer_master.csv", index=False)
    support_tickets.to_csv("data/processed/support_tickets.csv", index=False)
    print("Wrote data/processed/customer_master.csv")
    print("Wrote data/processed/support_tickets.csv")

    # --- Verification: confirm our 3 demo customers have both a customer_master
    # row AND a scenario ticket, so Explain will have real evidence to retrieve ---
    print("\n=== Verification: demo scenario customers have evidence ===")
    for t in DEMO_SCENARIO_TICKETS:
        cust = t["customer_id"]
        has_master = cust in customer_master["customer_id"].values
        has_ticket = cust in support_tickets[support_tickets["source"].str.contains("synthetic")]["customer_id"].values
        print(f"  {cust}: customer_master={'OK' if has_master else 'MISSING'}, "
              f"scenario_ticket={'OK' if has_ticket else 'MISSING'}")


if __name__ == "__main__":
    main()
