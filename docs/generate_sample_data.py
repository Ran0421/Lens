"""
Generates the small illustrative CSVs in data/sample/. Run from the repo
root: `python docs/generate_sample_data.py`

This is deterministic (seeded) and self-contained -- it does NOT hit
Kaggle. It exists so the sample data is reproducible and inspectable,
not a black box someone has to trust.
"""

import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
from acquire_data import build_customer_master, build_support_tickets

rng = np.random.default_rng(7)

REGIONS = ["East", "West", "Central", "South"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]

N_CUSTOMERS = 15
CUSTOMER_IDS = [f"CUST-{i:03d}" for i in range(1, N_CUSTOMERS + 1)]

# Force 3 customers into East/Corporate so the demo anomaly scenario is
# guaranteed reproducible at small sample size rather than left to chance.
FORCED_EAST_CORP = CUSTOMER_IDS[:3]


def build_transactions():
    customer_region = {c: "East" for c in FORCED_EAST_CORP}
    customer_segment = {c: "Corporate" for c in FORCED_EAST_CORP}
    for c in CUSTOMER_IDS[3:]:
        customer_region[c] = rng.choice(REGIONS)
        customer_segment[c] = rng.choice(SEGMENTS)

    baseline_weeks = pd.date_range("2026-01-05", "2026-08-03", freq="7D")
    anomaly_weeks = pd.date_range("2026-08-10", "2026-08-24", freq="7D")
    all_weeks = list(baseline_weeks) + list(anomaly_weeks)

    rows = []
    order_num = 1

    for d in all_weeks:
        for _ in range(rng.integers(2, 4)):
            cust = rng.choice(CUSTOMER_IDS)
            base_sales = rng.uniform(80, 900)
            if d in anomaly_weeks and customer_region[cust] == "East" and customer_segment[cust] == "Corporate":
                base_sales *= 0.55
                discount = rng.uniform(0.05, 0.10)
            else:
                discount = rng.uniform(0.0, 0.25)
            rows.append({
                "customer_id": cust, "region": customer_region[cust], "segment": customer_segment[cust],
                "category": rng.choice(CATEGORIES), "order_date": d, "order_id": f"ORD-{order_num:04d}",
                "sales": round(base_sales, 2), "discount": round(discount, 3),
                "profit": round(base_sales * rng.uniform(0.05, 0.30), 2),
            })
            order_num += 1

    # Guarantee East/Corporate presence every week (healthy baseline, clear dip)
    for d in all_weeks:
        for cust in FORCED_EAST_CORP:
            base_sales = rng.uniform(300, 500)
            if d in anomaly_weeks:
                base_sales *= 0.50
                discount = rng.uniform(0.05, 0.08)
            else:
                discount = rng.uniform(0.10, 0.20)
            rows.append({
                "customer_id": cust, "region": "East", "segment": "Corporate",
                "category": rng.choice(CATEGORIES), "order_date": d, "order_id": f"ORD-{order_num:04d}",
                "sales": round(base_sales, 2), "discount": round(discount, 3),
                "profit": round(base_sales * rng.uniform(0.10, 0.25), 2),
            })
            order_num += 1

    return pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)


def main():
    transactions = build_transactions()
    transactions.to_csv("data/sample/transactions_sample.csv", index=False)

    telco_like = pd.DataFrame({
        "tenure": rng.integers(1, 60, size=200),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=200, p=[0.55, 0.25, 0.20]),
        "MonthlyCharges": rng.uniform(20, 120, size=200).round(2),
        "Churn": rng.choice(["Yes", "No"], size=200, p=[0.27, 0.73]),
    })
    customer_master = build_customer_master(transactions, telco_like, as_of_week="2026-08-24", seed=7)
    customer_master.to_csv("data/sample/customer_master_sample.csv", index=False)

    ticket_texts = pd.DataFrame({"Ticket Description": [
        "Customer complained about the recent price increase on renewal, considering switching providers.",
        "Reported that the new pricing tier feels expensive compared to last quarter's plan.",
        "Login issue resolved after password reset, no further action needed.",
        "Requested refund due to shipping delay of 2 weeks beyond estimated date.",
        "Asked about bulk order discount for upcoming corporate renewal.",
        "Complained that support response time has gotten slower this month.",
        "Positive feedback on new product feature, no issue reported.",
        "Flagged a billing discrepancy between invoice and quoted price.",
        "Requested cancellation, citing better pricing from a competitor.",
        "Reported a defective item in last shipment, replacement requested.",
    ]})
    tickets = build_support_tickets(transactions, ticket_texts, tickets_per_customer_range=(0, 3), seed=7)

    # Force matching pricing-complaint tickets into the anomaly window for
    # the East/Corporate customers, so Explain-stage evidence is present.
    scenario_rows = [{
        "customer_id": cust, "region": "East", "segment": "Corporate",
        "ticket_date": pd.Timestamp("2026-08-14"),
        "ticket_text": "Customer complained about the recent price increase on renewal, considering switching providers.",
        "source": "support_tickets_scenario_synthetic", "grain": "event",
        "attribute_provenance": "fully synthetic, written for demo scenario",
    } for cust in FORCED_EAST_CORP]
    tickets = pd.concat([tickets, pd.DataFrame(scenario_rows)], ignore_index=True)
    tickets.to_csv("data/sample/support_tickets_sample.csv", index=False)

    print(f"transactions_sample.csv: {len(transactions)} rows")
    print(f"customer_master_sample.csv: {len(customer_master)} rows")
    print(f"support_tickets_sample.csv: {len(tickets)} rows")


if __name__ == "__main__":
    main()
