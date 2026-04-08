---
title: OpenEnv Tabular Cleaning
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Tabular Data Cleaning Environment

This is a real-world OpenEnv environment representing data engineering and standardization workflows. 
It utilizes a FastAPI backend implementing the OpenEnv specification (`/reset`, `/step`, `/state`) using Pydantic models.

## Environment Description & Motivation
**Domain:** Data Cleaning & Standardization.
**Motivation:** AI agents are frequently asked to clean, standardize, and aggregate tabular data. This environment simulates a headless data engineering system where an agent receives raw data previews and must iteratively issue exact declarative operations (like `FILL_MISSING`, `CONVERT_CURRENCY`, `GROUPBY_SUM`) to meet a hidden ground-truth requirement.

## Action Space
The Action space defines what transformations the agent can perform. It is a strictly typed JSON object:
- `op`: Enum specifying the operation (`FILL_MISSING`, `NORMALIZE_CASE`, `STRIP_WHITESPACE`, `DEDUP_ROWS`, `PARSE_DATE`, `RENAME_COLUMN`, `DROP_COLUMN`, `EXTRACT_MONTH`, `CONVERT_CURRENCY`, `GROUPBY_SUM`).
- `column`: String specifying the target column name.
- `value`: Optional string specifying the literal value, replacement, or new column name.
- `pattern`: Optional format string for dates.

## Observation Space
The Observation space returns the current metrics and previews of the dataset at each step:
- `task_id`: Current task tier (`easy`, `medium`, `hard`).
- `instructions`: Natural language task description.
- `preview_original`: First 3 rows of the raw data.
- `preview_current`: First 3 rows of the data after the most recent transformation.
- `cell_accuracy`: Float metric.
- `duplicate_rate`: Float metric.
- `missing_rate`: Float metric.
- `step_index`: Integer current step.

## Task Descriptions & Difficulty
1. **Easy (Single-column update)**: Clean whitespaces, fix case normalization, and fill missing data on a 'city' column.
2. **Medium (Multi-column cleaning)**: Deduplicate rows and standardize string/date formats on multiple columns.
3. **Hard (Aggregations and Conversions)**: Construct an end-to-end grouping pipeline, converting currencies (`EUR` to `USD`), isolating months from timestamps, and extracting totals using a `GROUPBY_SUM`.

## Setup and Usage Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the environment locally (uses port 7860):
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 7860
   ```
3. Alternately, use Docker:
   ```bash
   docker build -t data-cleaner .
   docker run -p 7860:7860 data-cleaner
   ```
4. Run the baseline agent script:
   ```bash
   export OPENAI_API_KEY="your_actual_key"
   python inference.py
   ```

## Baseline Scores
- **Easy**: ~0.95 (GPT-4o)
- **Medium**: ~0.88 (GPT-4o)
- **Hard**: ~0.84 (GPT-4o)
