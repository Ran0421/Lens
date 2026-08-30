# Lens — KPI Intelligence-to-Action Engine

**Accenture Innovation Challenge 2026 — Round 2 — Track: BusinessIntelligence.ai**
Team Neo — Ranjeeta Mashal (IIT Kharagpur, Metallurgical & Materials Engineering) · Rohini Nanaji Chavan (IIT Kharagpur, Chemical Engineering)

Lens detects material KPI movements, reconciles evidence across heterogeneous sources, ranks explanatory drivers, generates persona-specific narratives, communicates uncertainty honestly, and recommends grounded actions — with an explicit, auditable line between what a deterministic algorithm decided and what an LLM was allowed to phrase.

---

## Quickstart

```bash
git clone https://github.com/Ran0421/Lens.git
cd Lens
pip install -r requirements.txt

# 1. Prepare the real Superstore data (standardizes columns + documented injection)
python3 scripts/prepare_transactions.py

# 2. Build customer/ticket evidence sources (real Telco + real ticket data, remapped to real Superstore customers)
python3 scripts/build_evidence_sources.py

# 3. Run the deterministic pipeline (no API key needed for these three)
python3 src/detect.py
python3 src/decompose.py
python3 src/evidence_score.py

# 4. Run the LLM-touching stages (needs a free Groq API key — see below)
echo "GROQ_API_KEY=gsk_your_key_here" > .env
python3 src/explain.py
python3 src/recommend.py
python3 src/personas.py
python3 src/governance.py
python3 src/llm_eval.py

# 5. Start the live API + frontend
python3 -m uvicorn api.main:app --reload --port 8000
# open http://localhost:8000
```

A free Groq API key is available at [console.groq.com](https://console.groq.com).

---

## Architecture

```
Superstore (real, 2014-2017)  ─┐
Telco Churn (real)             ├──▶ Reconcile ──▶ Detect ──▶ Decompose ──▶ Explain ──▶ Recommend ──▶ Personas ──▶ Governance
Support Tickets (real)        ─┘         │            │            │            │            │            │            │
                                      non-LLM      non-LLM      non-LLM       LLM +      hybrid        LLM        non-LLM
                                                                            non-LLM
                                                                          fact-check
```

### LLM vs. non-LLM — the required breakdown

| Stage | What it does | Powered by |
|---|---|---|
| Reconcile | Standardizes real Superstore columns, aligns grains | Non-LLM — pandas |
| Detect | Seasonal-baseline anomaly flagging (median/MAD + business materiality, volume-eligibility guard) | Non-LLM — statistics |
| Decompose | Volume / mix / price waterfall decomposition; category contribution | Non-LLM — deterministic arithmetic |
| Evidence scoring | Evidence sufficiency (gates confidence) + attribution clarity (shapes narrative) | Non-LLM — weakest-link scoring |
| Explain | Ranks candidate causes; adversarial challenge; fact-checks the challenge | **LLM** (propose + challenge) + **non-LLM** (fact-check, confidence forced by evidence score) |
| Recommend | Driver → lever → owner mapping; action/impact phrasing | **Hybrid** — lookup table is deterministic, phrasing is LLM |
| Personas | Masks PII pre-prompt (non-LLM); phrases persona-specific narrative | **Non-LLM masking** + **LLM narrative** |
| Governance | RBAC access gate, checked *before* any evidence is built; append-only audit log | Non-LLM |
| LLM eval | Citation groundedness, confidence divergence, PII leak scan, numeric consistency, driver alignment | Non-LLM (reads existing outputs, no new API calls) |

**The LLM is never the source of quantitative truth.** Every number an LLM narrative states is traceable to a field computed upstream by deterministic code — the eval suite (`src/llm_eval.py`) checks this automatically and has already caught one real distortion bug (see Known Issues Found and Fixed, below).

---

## Data: real, with one documented, minimal, disclosed injection

- **Transactions**: real Kaggle Superstore dataset (9,994 rows, 793 customers, 2014–2017).
- **Customer master**: every row's `customer_id` is a real Superstore customer. Attributes (tenure, contract type, churn flag) are sampled from the real Telco Churn dataset's *distribution* — not copied from Telco's own (unrelated) customers.
- **Support tickets**: real ticket text from the real Customer Support Ticket dataset, remapped to real Superstore customers and real order windows. The dataset's `{product_purchased}` template placeholder (present in 100% of source rows) is filled with each ticket's actual matched product.
- **The one injection**: 3 real rows (East/Corporate, week of 2017-12-11) have `sales`/`profit` scaled by 1.4x to complete a multi-factor demo scenario. Full before/after values are in `docs/injection_log.md`. Everything else — all 9,991 other rows, every other week, every other region/segment — is completely untouched real data.

---

## The 4 required demo scenarios

| # | Scenario | Where it lives | Slice |
|---|---|---|---|
| 1 | Multi-factor movement | Explain/Recommend/Personas | Revenue, East/Corporate, week of 2017-12-11 |
| 2 | Low-confidence/abstention | Explain/Recommend/Personas | Revenue, Central/Consumer, week of 2017-12-25 (-88%, zero supporting tickets) |
| 3 | Sparse-history | Explain (deterministic path, no LLM call) | Revenue, South/Home Office, week of 2014-01-06 (first-ever observation, 0 baseline weeks) |
| 4 | Role-based security | Governance | Regional Manager West denied East; Finance VP sees all regions, masked |

---

## Known issues found and fixed (kept here deliberately, as evidence of real evaluation)

- **Groq model deprecation**: the originally-planned model (`llama-3.1-8b-instant`) stopped being available on the account between planning and execution. Fixed by querying the live `/models` endpoint directly and switching to `openai/gpt-oss-20b`. `GROQ_MODEL` is an environment variable specifically so this can be swapped without a code change if it happens again.
- **Unit-distortion bug**: a Finance VP narrative correctly cited a real figure (`$1,309.23`) but appended a spurious "k" (thousands) multiplier, inflating it 1000x. Caught by `src/llm_eval.py`'s numeric consistency check, which extracts every dollar figure from every LLM output and verifies it against real evidence-packet fields. Documented in `llm_eval.py`'s docstring as the reason that check exists.
- **Confidence divergence (a finding, not a bug)**: for the low-confidence scenario, the LLM's own self-reported confidence was "high" — the deterministic evidence gate correctly forced it down to `low_confidence_abstain`. This is the concrete, measured justification for the hybrid confidence rule, not a hypothetical.

---

## Repo structure

```
Lens/
├── scripts/
│   ├── prepare_transactions.py    # standardizes + injects (documented) the real Superstore data
│   └── build_evidence_sources.py  # builds customer_master + support_tickets from real data
├── src/
│   ├── acquire_data.py            # customer_master / support_tickets construction logic
│   ├── reconcile.py                # SQL-style reconciliation of daily/weekly grains
│   ├── detect.py                   # anomaly detection (median/MAD, volume-eligibility guard)
│   ├── decompose.py                # volume/mix/price waterfall + category breakdown
│   ├── evidence_score.py           # evidence sufficiency + attribution clarity scoring
│   ├── explain.py                  # propose/challenge/fact-check via Groq
│   ├── recommend.py                # driver->lever mapping + grounded action phrasing
│   ├── personas.py                 # PII masking + persona-specific narrative generation
│   ├── governance.py                # RBAC access gate + audit log
│   └── llm_eval.py                  # 6 deterministic checks against all LLM outputs
├── api/
│   └── main.py                      # real FastAPI wiring: dashboard, investigate, history, feedback(stub)
├── frontend/
│   └── index.html                   # single-file frontend (no build step), real API calls
├── data/
│   ├── raw/                          # real Superstore/Telco/ticket CSVs (gitignored except superstore_raw.csv)
│   └── processed/                    # all pipeline outputs (gitignored — regenerate via the scripts above)
├── docs/
│   └── injection_log.md              # exact rows/values changed by the one documented injection
└── requirements.txt
```

---

## What's not yet built

- **`feedback.py`** — the lightweight closed-loop design (analyst confirms/rejects a cause, few-shot retrieval on similar future cases, per-cause-type reliability scoring) is designed but not implemented. `/api/feedback` is currently an in-memory stub.
- **Business Proposal PDF/PPT** — the written deliverable is separate from this technical prototype.
- **Test fixtures** — the 4 demo scenarios are runnable via the scripts above but aren't yet packaged as `pytest` fixtures under `tests/scenarios/`.

---

## Setup notes

- `data/processed/` is regenerated by the scripts above and is gitignored — don't expect it to be present after a fresh clone.
- `.env` (holding `GROQ_API_KEY`) is gitignored and must be created locally — never commit it.
- Groq's free tier has a per-minute token limit; `explain.py`'s `call_groq()` automatically retries once on a 429, respecting Groq's own suggested wait time from the error message.
