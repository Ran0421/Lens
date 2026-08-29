# Injection Log - Lens Round 2

This file documents the ONLY synthetic modification made to the real
Superstore dataset used in this prototype. All other 9,991 rows are
100% real, unmodified Kaggle Superstore data.

**Scope:** region=`East`, segment=`Corporate`, week starting `2017-12-11`

**Change:** `sales` and `profit` multiplied by **1.4** on the rows below (margin ratio preserved).

**Why:** this week already had a real, unmodified volume drop (3 orders vs. ~5.5 baseline) and a real, unmodified discount pull-back (0.067 vs. ~0.165 baseline mean). This injection adds the third lever — a price increase — to complete an honest multi-factor revenue-decline scenario for the Detect/Decompose/Explain pipeline.

## Rows modified

| order_id       | customer_id   | order_date   | category        |   sales_before |   sales_after |   profit_before |   profit_after |   multiplier |
|:---------------|:--------------|:-------------|:----------------|---------------:|--------------:|----------------:|---------------:|-------------:|
| CA-2017-129581 | KN-16390      | 2017-12-11   | Technology      |        128.85  |       180.39  |          3.8655 |         5.4117 |          1.4 |
| CA-2017-135377 | BP-11095      | 2017-12-13   | Furniture       |        287.976 |       403.166 |          7.1994 |        10.0792 |          1.4 |
| US-2017-125213 | NB-18655      | 2017-12-11   | Office Supplies |          6.54  |         9.156 |          2.1582 |         3.0215 |          1.4 |

