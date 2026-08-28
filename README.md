# Lens - Round 2 (BusinessIntelligence.ai)

A KPI intelligence-to-action engine. See `docs/architecture.md` (coming next)
for the full 8-requirement mapping and the LLM vs. non-LLM breakdown.

## Status

**Built:**
- [x] 5 KPI semantic contracts (`contracts/*.yaml`) - definitions, thresholds, lineage, access rules
- [x] Data acquisition layer (`src/acquire_data.py`) - 3 sources, real primary-key integrity (see docstrings)
- [x] Reconcile layer (`src/reconcile.py`) - DuckDB warehouse, grain alignment, freshness tagging
- [x] Sample data (`data/sample/`) - with a guaranteed, verified anomaly scenario
- [x] Detect stage (`src/detect.py`) - seasonal baseline + statistical/business materiality (OR combine rule, documented tradeoff). Tested against sample data: correctly flags the known East/Corporate anomaly. See `docs/detect_threshold_tuning.md` for why sample-scale testing uses a scoped threshold override (contracts themselves are untouched).
- [x] Replit-ready API stub (`api/main.py`) + frontend (`frontend/index.html`)

**Not yet built:**
- [ ] Decompose (variance/contribution decomposition)
- [ ] Explain (evidence retrieval + LLM cause ranking + adversarial challenge + confidence gate)
- [ ] Recommend (lever lookup + LLM phrasing)
- [ ] Personas + narrative templates
- [ ] Governance (RBAC, masking, audit log)
- [ ] Feedback loop (lightweight closed loop)
- [ ] Telemetry decorator
- [ ] Real pipeline wired into FastAPI (currently hardcoded stub responses)
- [ ] 4 required scenarios as test fixtures
- [ ] Frontend fully wired to real pipeline output

## Data sources

| Source | Dataset | Grain |
|---|---|---|
| A. Transactions | [Superstore](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) | Daily |
| B. Customer master | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Weekly |
| C. Support tickets | [Customer Support Tickets](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) | Event-level |

**Documented assumption:** none of these three datasets share a real foreign
key (no public dataset does, for this exact combination). Source B is mapped
onto Source A's Region dimension via a fixed seeded random assignment; ticket
data is used for realistic structure/format, with scenario-specific synthetic
tickets overlaid for the 4 required demo scenarios. This is disclosed here
and in `src/acquire_data.py` docstrings — the brief explicitly does not
expect real proprietary joined data.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
python -c "from src.acquire_data import download_sources; download_sources()"
```

## Repo structure

```
lens-round2/
├── contracts/          # KPI semantic contracts (YAML)
├── src/
│   ├── acquire_data.py # source loading + standardization
│   └── reconcile.py    # DuckDB warehouse, grain alignment, freshness
├── data/                # raw + processed (gitignored)
├── api/                 # FastAPI app (not yet built)
├── notebooks/           # end-to-end walkthrough (not yet built)
├── frontend/             # dashboard (not yet built)
├── tests/scenarios/      # required scenario fixtures (not yet built)
└── docs/                  # architecture doc (not yet built)
```
