---
title: OpenEnv Tabular Cleaning
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧹 OpenEnv: Tabular Data Cleaner

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/CodyRohith7/OpenEnv-Tabular-cleaning)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, deterministic OpenEnv environment designed for the **Meta PyTorch Hackathon**. This environment simulates real-world data engineering tasks where an AI agent must iteratively clean, standardize, and aggregate raw tabular data to match a hidden ground-truth requirement.

## 🚀 Deployment
- **Hugging Face Space**: [Live Demo](https://huggingface.co/spaces/CodyRohith7/OpenEnv-Tabular-cleaning)
- **API Endpoint**: `https://codyrohith7-openenv-tabular-cleaning.hf.space`

## 🧠 Environment Design
The environment implements the **OpenEnv Specification**, providing a strictly-typed API for agentic interaction.

### Task Tiers
| Tier | Description | Key Operations |
| :--- | :--- | :--- |
| **Easy** | Single-column normalization | `STRIP_WHITESPACE`, `NORMALIZE_CASE`, `FILL_MISSING` |
| **Medium** | Multi-column standardizing | `PARSE_DATE`, `DEDUP_ROWS`, `DROP_COLUMN` |
| **Hard** | Complex Data Engineering | `CONVERT_CURRENCY`, `EXTRACT_MONTH`, `GROUPBY_SUM` |

### Action Space (`/step`)
Agents interact by sending JSON actions:
- `op`: The operation to perform (e.g., `FILL_MISSING`).
- `column`: Target column name.
- `value`: Optional parameter (e.g., new column name or fill value).

### Observation Space
- `preview_current`: Real-time view of the top 3 rows.
- `cell_accuracy`: Percentage match with ground truth.
- `missing_rate`: Current ratio of NaNs.
- `duplicate_rate`: Ratio of redundant rows.

## 🛠️ Local Setup

1. **Clone the Repo**
   ```bash
   git clone https://github.com/CodyRohith7/OpenEnv_Tabular_DataCleaner
   cd OpenEnv_Tabular_DataCleaner
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Locally**
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 7860
   ```

## 🤖 Baseline Evaluation
To run the included GPT-4o baseline agent against your live environment:

```bash
export OPENAI_API_KEY="your_key"
export ENV_URL="https://codyrohith7-openenv-tabular-cleaning.hf.space"
python inference.py
```

## ⚖️ Evaluation Metric
The final score is a composite of **Cell Accuracy**, **Schema Correctness**, and **Deduplication F1-Score**, ensuring the agent doesn't just "guess" but actually structures the data correctly.

---
Built with ❤️ for the **Meta PyTorch Hackathon 2026**.
