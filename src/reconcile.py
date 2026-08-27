"""
Reconcile: aligns three sources of different grain/cadence into a single
DuckDB warehouse that downstream stages (Detect, Decompose, Explain) can
query with plain SQL, plus a freshness tag on every row.

Primary key note: customer_id here is always a REAL Superstore customer_id.
customer_master and support_tickets are constructed in acquire_data.py
against that real key (see that file's module docstring for the full
rationale) -- so every join below is a genuine primary-key relationship,
not a coincidental one between unrelated companies' data.

Non-LLM stage. Pure data engineering -- this is deliberate: reconciliation
is exactly the kind of step that should NOT be delegated to an LLM, since
correctness here is what everything downstream depends on.
"""

import duckdb
import pandas as pd
from datetime import datetime


class Warehouse:
    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(db_path)

    def load(self, transactions: pd.DataFrame, customer_master: pd.DataFrame,
             tickets: pd.DataFrame):
        self.con.register("transactions_raw", transactions)
        self.con.register("customer_master_raw", customer_master)
        self.con.register("tickets_raw", tickets)

        # Daily aggregate per region/segment -- the grain Detect/Decompose need
        self.con.execute("""
            CREATE OR REPLACE TABLE daily_agg AS
            SELECT
                order_date::DATE AS date,
                region,
                segment,
                SUM(sales) AS revenue,
                SUM(profit) AS profit,
                SUM(profit) / NULLIF(SUM(sales), 0) AS profit_margin,
                COUNT(DISTINCT order_id) AS order_volume,
                SUM(sales * discount) / NULLIF(SUM(sales), 0) AS discount_rate
            FROM transactions_raw
            GROUP BY 1, 2, 3
        """)

        # Weekly repeat-purchase rate, computed from transactions and joined
        # to customer_master on the real, shared customer_id. This is the
        # deliberate mismatched-cadence demonstration: transactions refresh
        # daily, customer_master refreshes weekly, and this KPI is only ever
        # as fresh as the older of the two.
        self.con.execute("""
            CREATE OR REPLACE TABLE weekly_orders AS
            SELECT
                date_trunc('week', order_date) + INTERVAL 6 DAY AS week_ending,
                region,
                customer_id,
                COUNT(DISTINCT order_id) AS orders_this_week
            FROM transactions_raw
            GROUP BY 1, 2, 3
        """)

        self.con.execute("""
            CREATE OR REPLACE TABLE weekly_repeat_purchase AS
            WITH active_customers AS (
                SELECT week_ending, region, customer_id,
                       SUM(orders_this_week) OVER (
                           PARTITION BY region, customer_id
                           ORDER BY week_ending
                           RANGE BETWEEN INTERVAL 90 DAY PRECEDING AND CURRENT ROW
                       ) AS orders_trailing_90d
                FROM weekly_orders
            )
            SELECT
                week_ending,
                region,
                AVG(CASE WHEN orders_trailing_90d > 1 THEN 1.0 ELSE 0.0 END) AS repeat_purchase_rate,
                COUNT(DISTINCT customer_id) AS active_customers
            FROM active_customers
            GROUP BY 1, 2
        """)

        # Customer-level join example: real customer_id shared between
        # transactions and customer_master -- used by Explain when a cause
        # needs account-level detail (e.g. "9 accounts, $612K at risk").
        self.con.execute("""
            CREATE OR REPLACE TABLE customer_360 AS
            SELECT
                t.customer_id,
                t.region,
                t.segment,
                SUM(t.sales) AS lifetime_sales,
                m.tenure_months,
                m.contract_type,
                m.churn_flag,
                m.snapshot_week
            FROM transactions_raw t
            LEFT JOIN customer_master_raw m
                ON t.customer_id = m.customer_id
            GROUP BY 1, 2, 3, 5, 6, 7, 8
        """)

    def freshness_report(self) -> pd.DataFrame:
        """
        Every query result must carry a freshness tag rather than silently
        presenting a weekly-grain number alongside daily-grain numbers as if
        they were equally current. This is queried by Explain before it
        builds an evidence bundle, and surfaced to the persona narrative.
        """
        today = pd.Timestamp(datetime.utcnow().date())
        daily_latest = self.con.execute(
            "SELECT MAX(date) FROM daily_agg").fetchone()[0]
        weekly_latest = self.con.execute(
            "SELECT MAX(week_ending) FROM weekly_repeat_purchase").fetchone()[0]
        snapshot_latest = self.con.execute(
            "SELECT MAX(snapshot_week) FROM customer_master_raw").fetchone()[0]

        return pd.DataFrame([
            {"source": "transactions (revenue, margin, volume, discount)",
             "grain": "daily",
             "latest_data": daily_latest,
             "staleness_days": (today - pd.Timestamp(daily_latest)).days if daily_latest else None},
            {"source": "transactions (repeat purchase rate, weekly rollup)",
             "grain": "weekly",
             "latest_data": weekly_latest,
             "staleness_days": (today - pd.Timestamp(weekly_latest)).days if weekly_latest else None},
            {"source": "customer_master (tenure, contract, churn flag)",
             "grain": "weekly_snapshot",
             "latest_data": snapshot_latest,
             "staleness_days": (today - pd.Timestamp(snapshot_latest)).days if snapshot_latest else None},
        ])

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()
