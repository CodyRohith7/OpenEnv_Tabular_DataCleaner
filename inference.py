"""
OpenEnv Baseline Inference Script
===================================
Runs an LLM agent against the Enterprise Data Cleaning Environment
across all four tasks (easy → medium → hard → extreme) and prints
structured output for the OpenEnv validator.

Usage:
    export OPENAI_API_KEY="sk-..."         # required for real LLM run
    export API_BASE_URL="https://..."      # optional, defaults to OpenAI
    export MODEL_NAME="gpt-4o-mini"       # optional
    export ENV_URL="http://localhost:7860" # optional
    python inference.py
"""

import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from typing import List, Dict, Any


# ─── Structured output helpers expected by openenv validate ────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={reward:.4f}", flush=True)


def log_end(task: str, score: float, steps: int) -> None:
    print(f"[END] task={task} score={score:.4f} steps={steps}", flush=True)


# ─── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Data Engineering AI agent operating inside an OpenEnv environment.
Your job: apply a sequence of declarative data-cleaning operations to
transform a raw tabular dataset into a clean, gold-standard output.

Respond with ONLY a valid JSON object — no explanations, no markdown.

Action schema:
{
  "op":      "<OpType>",         // required — one of the ops below
  "column":  "<col_name>",       // target column (comma-sep for multi-col ops)
  "value":   "<string>",         // replacement value / new column name / JSON key
  "pattern": "<string>"          // date format string OR target col for EXTRACT_JSON
}

Available operations:
  STRIP_WHITESPACE   – strip leading/trailing spaces from column
  NORMALIZE_CASE     – title-case all values in column
  FILL_MISSING       – fill NaN values with 'value' (default: "Unknown")
  CAST_NUMERIC       – parse string column to float64
  PARSE_DATE         – normalize dates to YYYY-MM-DD
  DEDUP_ROWS         – drop duplicate rows by key columns (comma-sep in column)
  DROP_COLUMN        – remove a column entirely
  RENAME_COLUMN      – rename column to 'value'
  EXTRACT_MONTH      – extract YYYY-MM from timestamp; output col name in 'value'
  CONVERT_CURRENCY   – normalize EUR→USD (×1.08) and GBP→USD (×1.27)
  GROUPBY_SUM        – group by columns (comma-sep), sum 'value' column
  PII_REDACT         – mask emails in column with [REDACTED]
  EXTRACT_JSON       – parse JSON string, extract key 'value' → new col 'pattern'
  DROP_OUTLIERS      – remove rows where column's z-score > 3

The instructions field in the observation tells you EXACTLY what to do and in
what order. Follow them precisely. One action per response.
"""


# ─── Environment client ────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.http = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def reset(self, task_id: str) -> Dict[str, Any]:
        r = await self.http.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    async def step(self, action: dict) -> Dict[str, Any]:
        r = await self.http.post("/step", json=action)
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self.http.aclose()


# ─── Task runner ───────────────────────────────────────────────────────────

MAX_STEPS = {"easy": 8, "medium": 10, "hard": 15, "extreme": 20}


async def run_task(
    task_id: str,
    client: EnvClient,
    llm: AsyncOpenAI,
    model_name: str,
) -> Dict[str, Any]:
    log_start(task=task_id)
    result = {"task": task_id, "score": 0.0, "steps": 0, "error": None}

    # ── Reset ──────────────────────────────────────────────────────────────
    try:
        obs = await client.reset(task_id)
    except Exception as e:
        result["error"] = str(e)
        log_end(task=task_id, score=0.001, steps=0)
        return result

    history: List[tuple] = []
    max_steps = MAX_STEPS.get(task_id, 15)
    done = False
    score = 0.0
    steps_taken = 0

    for step_num in range(1, max_steps + 1):
        if done:
            break

        # ── Build prompt ───────────────────────────────────────────────────
        prompt = (
            f"=== TASK: {obs['task_id'].upper()} ===\n"
            f"Instructions:\n{obs['instructions']}\n\n"
            f"Current columns: {obs.get('columns', [])}\n"
            f"Column types:    {obs.get('column_types', {})}\n"
            f"Row count:       {obs.get('row_count', '?')}\n"
            f"Cell accuracy:   {obs['cell_accuracy']:.4f}\n"
            f"Schema score:    {obs.get('schema_score', 1.0):.4f}\n"
            f"Missing rate:    {obs['missing_rate']:.4f}\n"
            f"Duplicate rate:  {obs['duplicate_rate']:.4f}\n"
            f"Step:            {obs['step_index']}/{max_steps}\n\n"
            f"Preview (current):\n{json.dumps(obs['preview_current'], indent=2)}\n\n"
            "Output the next single JSON action:"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, asst_msg in history[-6:]:          # last 6 turns context
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})
        messages.append({"role": "user", "content": prompt})

        # ── LLM call ───────────────────────────────────────────────────────
        try:
            response = await llm.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            action_text = response.choices[0].message.content
            action = json.loads(action_text)
        except Exception as e:
            result["error"] = f"LLM error @ step {step_num}: {e}"
            log_step(step=step_num, reward=0.0)
            done = True
            break

        history.append((prompt, action_text))

        # ── Environment step ───────────────────────────────────────────────
        try:
            step_result = await client.step(action)
            obs       = step_result["observation"]
            reward    = step_result["reward"]["value"]
            done      = step_result["done"]
            info      = step_result["info"]
        except Exception as e:
            result["error"] = f"Env error @ step {step_num}: {e}"
            log_step(step=step_num, reward=0.0)
            done = True
            break

        steps_taken = step_num
        log_step(step=step_num, reward=reward)

        if done:
            score = info.get("final_score", obs.get("cell_accuracy", 0.001))

    score = max(0.001, min(0.999, score))
    result.update({"score": score, "steps": steps_taken})
    log_end(task=task_id, score=score, steps=steps_taken)
    return result


# ─── Main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    api_key    = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-key"
    base_url   = os.getenv("API_BASE_URL",  "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME",    "gpt-4o-mini")
    env_url    = os.getenv("ENV_URL",       "http://localhost:7860")

    llm    = AsyncOpenAI(api_key=api_key, base_url=base_url)
    client = EnvClient(base_url=env_url)

    tasks   = ["easy", "medium", "hard", "extreme"]
    results: List[Dict] = []

    for task_id in tasks:
        r = await run_task(task_id, client, llm, model_name)
        results.append(r)

    await client.close()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*52, flush=True)
    print(f"{'TASK':<10} {'SCORE':>8} {'STEPS':>8} {'STATUS':>10}", flush=True)
    print("-"*52, flush=True)
    for r in results:
        status = "✓ OK" if r["error"] is None else "✗ ERR"
        print(f"{r['task']:<10} {r['score']:>8.4f} {r['steps']:>8} {status:>10}", flush=True)
    print("="*52, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
