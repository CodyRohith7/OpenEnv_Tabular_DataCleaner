<div align="center">

# 🧹 OpenEnv: Enterprise Data Cleaning Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-grade reinforcement learning environment that simulates real enterprise data engineering pipelines.**

An AI agent observes a raw, noisy tabular dataset and applies a sequence of declarative ETL operations to reconcile it against a hidden gold-standard — across four progressive difficulty tiers.

</div>

---

## 📖 Overview

Every data-driven company faces this problem daily: raw exports from CRMs, ERPs, and marketing platforms arrive **dirty** — wrong date formats, mixed-case fields, embedded JSON, PII, currency mismatches, and statistical outliers. A data engineer must apply a precise sequence of transformations to produce a clean analytics table.

This project models that exact pipeline as an **OpenEnv-compatible environment**, giving RL agents a rich, deterministic, partial-progress reward signal to learn from. The reward function guides the agent step-by-step — not just at episode end.

---

## 🏗️ Architecture

```
┌─────────────────────┐      POST /reset       ┌──────────────────────┐
│   Inference Agent   │ ─────────────────────▶ │   FastAPI Server     │
│  (LLM or custom)    │      POST /step        │   (server/main.py)   │
└─────────────────────┘ ◀───────────────────── └──────────┬───────────┘
                        Observation + Reward               │
                                                  DataCleaningEnv
                                                  (environment.py)
                                                           │
                                              ┌────────────▼───────────┐
                                              │   Pandas ETL Engine    │
                                              │  cell_accuracy         │
                                              │  dedup_F1              │
                                              │  schema_score          │
                                              └────────────────────────┘
```

The root endpoint (`/`) serves a **Glassmorphism dark-mode dashboard** where you can watch the baseline agent clean data live, with animated KPI cards and a reward-per-step chart.

---

## 🎯 Tasks

| ID | Difficulty | Domain | Key Operations | Max Steps | Rows |
|---|---|---|---|---|---|
| `easy` | 🟢 Easy | CRM Contact Export | STRIP_WHITESPACE, NORMALIZE_CASE, FILL_MISSING | 8 | 20 |
| `medium` | 🟡 Medium | Sales Order Reconciliation | PARSE_DATE (mixed formats), NORMALIZE_CASE, DEDUP_ROWS | 10 | 30 |
| `hard` | 🔴 Hard | Multi-Currency Revenue Ledger | EXTRACT_MONTH, CONVERT_CURRENCY, GROUPBY_SUM, RENAME_COLUMN | 15 | 47 |
| `extreme` | 🟣 Extreme | Full ETL Pipeline | EXTRACT_JSON, CAST_NUMERIC, PII_REDACT, DROP_OUTLIERS, NORMALIZE_CASE, GROUPBY_SUM, RENAME_COLUMN | 20 | 34 |

### Task Descriptions

**Easy — CRM Contact Normalization**
Clean a CRM export of 20 contact records. Apply whitespace stripping, Title Case normalization to `full_name` and `region` columns, then impute 6 missing region values with `"Unknown"`.

**Medium — Sales Order Standardization**
Standardize 30 order records (including 8 true duplicates). Normalize country names to Title Case, parse 3 different date formats to `YYYY-MM-DD`, and remove duplicate `(order_id, line_item)` pairs.

**Hard — Multi-Currency Revenue Ledger**
Transform 47 transaction rows spanning 5 months and 4 currencies. Extract month periods, convert EUR→USD (×1.08) and GBP→USD (×1.27), aggregate with GROUPBY_SUM, and rename the resulting column.

**Extreme — Full ETL Pipeline**
A raw marketing lead scrape of 34 rows containing embedded JSON strings, PII email addresses, mixed-case region labels, and 2 statistical revenue outliers (z-score > 3). The agent must parse JSON, cast types, redact PII, drop outliers, normalize, aggregate, and rename — a complete 7-step transformation chain.

---

## 🔧 Action Space

| Operation | `column` | `value` | `pattern` | Description |
|---|---|---|---|---|
| `STRIP_WHITESPACE` | target col | — | — | Remove leading/trailing spaces |
| `NORMALIZE_CASE` | target col | — | — | Convert values to Title Case |
| `FILL_MISSING` | target col | fill value | — | Replace NaN with `value` |
| `CAST_NUMERIC` | target col | — | — | Parse strings to float64 |
| `PARSE_DATE` | date col | — | — | Normalize to `YYYY-MM-DD` |
| `DEDUP_ROWS` | key cols (comma-sep) | — | — | Drop duplicate rows by key |
| `DROP_COLUMN` | col to drop | — | — | Remove column entirely |
| `RENAME_COLUMN` | old name | new name | — | Rename a column |
| `EXTRACT_MONTH` | timestamp col | new col name | — | Extract `YYYY-MM` period |
| `CONVERT_CURRENCY` | currency col | amount col | — | EUR→USD (×1.08), GBP→USD (×1.27) |
| `GROUPBY_SUM` | group cols (comma-sep) | value col | — | Group and sum |
| `PII_REDACT` | email col | — | — | Mask emails with `[REDACTED]` |
| `EXTRACT_JSON` | JSON string col | JSON key | new col name | Parse JSON, extract key |
| `DROP_OUTLIERS` | numeric col | — | — | Remove rows with z-score > 3 |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task identifier |
| `instructions` | `str` | Step-by-step ETL instructions for the agent |
| `preview_original` | `List[RowPreview]` | First 5 rows of the original raw data |
| `preview_current` | `List[RowPreview]` | First 5 rows of the current transformed data |
| `columns` | `List[str]` | Current column names |
| `column_types` | `Dict[str, str]` | Dtype of each column (`"float64"`, `"object"`, etc.) |
| `row_count` | `int` | Current number of rows |
| `cell_accuracy` | `float [0,1]` | Composite progress score vs. gold dataset |
| `schema_score` | `float [0,1]` | Schema structure match against target |
| `missing_rate` | `float [0,1]` | Fraction of NaN cells |
| `duplicate_rate` | `float [0,1]` | Fraction of duplicate rows |
| `step_index` | `int` | Steps taken so far |
| `available_ops` | `List[str]` | All valid OpType values |

---

## 🏆 Reward Function

The reward signal provides **meaningful partial progress** throughout the episode:

| Situation | Reward |
|---|---|
| Valid action with improvement (delta > 0) | `min(1.0, 0.3 + delta × 5.0)` |
| Valid action, no change (no-op) | `0.05` |
| Invalid action (bad column name, etc.) | `max(0.0, 0.1 − penalty)` |
| Destructive action | `0.0` |
| **Efficiency bonus** (early completion) | `+0.05 × (steps_remaining / max_steps)` |

**Composite metric driving delta:**
- **Easy**: `cell_accuracy` (100%)
- **Medium**: `0.6 × cell_accuracy + 0.4 × dedup_F1`
- **Hard / Extreme**: `0.4 × cell_accuracy + 0.3 × dedup_F1 + 0.3 × schema_score`

---

## 📊 Baseline Results

*Agent: GPT-4o-mini at temperature=0, following explicit task instructions.*

| Task | Score | Steps Used | Notes |
|---|---|---|---|
| easy | 0.999 | 5 / 8 | Perfect normalization + efficiency bonus |
| medium | 0.999 | 4 / 10 | Perfect dedup + date parse + efficiency bonus |
| hard | 0.995 | 4 / 15 | Multi-currency aggregation, ~0.4% float rounding gap |
| extreme | 0.990 | 7 / 20 | Full 7-step pipeline, outlier removal confirmed |

---

## 🚀 Quick Start

### Option 1 — Local (Python)

```bash
# 1. Clone
git clone https://github.com/CodyRohith7/OpenEnv_Tabular_DataCleaner.git
cd OpenEnv_Tabular_DataCleaner

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload

# 5. Open in browser
# Dashboard:  http://localhost:7860
# API docs:   http://localhost:7860/docs
```

### Option 2 — Docker

```bash
# Build
docker build -t openenv-data-cleaner .

# Run
docker run -p 7860:7860 openenv-data-cleaner

# Verify
curl http://localhost:7860/health
```

---

## 🤖 Running the Baseline Inference Agent

The `inference.py` script runs an LLM-powered agent against all four tasks and prints structured output.

```bash
# Required: your OpenAI-compatible API key
export OPENAI_API_KEY="sk-..."

# Optional overrides
export MODEL_NAME="gpt-4o-mini"           # default: gpt-4o-mini
export API_BASE_URL="https://api.openai.com/v1"  # default: OpenAI
export ENV_URL="http://localhost:7860"    # default: localhost

python inference.py
```

Expected output:
```
[START] task=easy
[STEP] step=1 reward=0.8000
[STEP] step=2 reward=0.9500
...
[END] task=easy score=0.9990 steps=5
```

> **Note:** `inference.py` works with any OpenAI-compatible endpoint. You can substitute any provider by setting `API_BASE_URL` and `OPENAI_API_KEY` accordingly.

---

## 📁 Project Structure

```
OpenEnv_Tabular_DataCleaner/
├── server/
│   ├── main.py          # FastAPI app — /reset, /step, /health, /tasks
│   ├── environment.py   # DataCleaningEnv — ETL engine & reward logic
│   ├── models.py        # Pydantic models (Action, Observation, Reward)
│   ├── utils.py         # Metric helpers (cell_accuracy, dedup_F1, schema_score)
│   ├── datasets/        # Raw & gold CSV files for all 4 tasks
│   └── static/
│       └── index.html   # Glassmorphism interactive dashboard
├── inference.py         # LLM baseline agent (OpenAI-compatible)
├── openenv.yaml         # OpenEnv environment specification
├── Dockerfile           # Production container definition
├── requirements.txt     # Python dependencies
└── pyproject.toml       # Package metadata
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive dashboard (HTML) |
| `GET` | `/health` | Health check — returns version & task list |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/tasks/{id}/gold-preview` | First 10 rows of the gold dataset |
| `POST` | `/reset` | Reset environment for a task; returns initial observation |
| `POST` | `/step` | Apply one action; returns observation, reward, done, info |
| `GET` | `/state` | Current episode state (episode_id, step, score) |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### Example: Reset + Step

```bash
# Reset to the easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Apply an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"op": "STRIP_WHITESPACE", "column": "full_name"}'
```

---

## 🛠️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for inference)* | API key for LLM calls |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use for the inference agent |
| `ENV_URL` | `http://localhost:7860` | URL of the running environment server |

> **Security:** Never commit secrets to version control. Use environment variables or a `.env` file (already in `.gitignore`).

---

## 📄 License

MIT © [CodyRohith7](https://github.com/CodyRohith7)
