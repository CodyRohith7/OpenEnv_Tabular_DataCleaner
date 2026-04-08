import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from typing import List

def log_start(task: str):
    print(f"[START] task={task}", flush=True)

def log_step(step: int, reward: float):
    print(f"[STEP] step={step} reward={reward:.2f}", flush=True)

def log_end(task: str, score: float, steps: int):
    print(f"[END] task={task} score={score:.3f} steps={steps}", flush=True)

MAX_STEPS = {"easy": 5, "medium": 10, "hard": 15}

class EnvClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def reset(self, task_id: str):
        res = await self.client.post("/reset", json={"task_id": task_id})
        res.raise_for_status()
        return res.json()

    async def step(self, action: dict):
        res = await self.client.post("/step", json=action)
        res.raise_for_status()
        return res.json()

async def run_task(task_id: str, client: EnvClient, llm: AsyncOpenAI, model_name: str):
    log_start(task=task_id)
    
    try:
        obs = await client.reset(task_id)
    except Exception as e:
        print(f"Error resetting environment: {e}", flush=True)
        log_end(task=task_id, score=0.0, steps=0)
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    
    system_prompt = (
        "You are a helpful Data Engineer agent. "
        "Complete the tabular data cleaning task below.\n\n"
        "You can output ONLY a valid JSON action matching this schema:\n"
        "{\n"
        '  "op": "FILL_MISSING" | "NORMALIZE_CASE" | "STRIP_WHITESPACE" | "DEDUP_ROWS" | "PARSE_DATE" | "RENAME_COLUMN" | "DROP_COLUMN" | "EXTRACT_MONTH" | "CONVERT_CURRENCY" | "GROUPBY_SUM",\n'
        '  "column": "column_name" (optional),\n'
        '  "value": "string" (optional)\n'
        "}\n\n"
        "Read the preview and instructions closely."
    )
    
    max_steps = MAX_STEPS.get(task_id, 10)
    done = False
    
    for step in range(1, max_steps + 1):
        if done:
            break

        prompt = (
            f"Instructions: {obs['instructions']}\n"
            f"Preview: {json.dumps(obs['preview_current'])}\n"
            f"Metrics: missing={obs['missing_rate']}, dup={obs['duplicate_rate']}, accuracy={obs['cell_accuracy']}\n"
            "Produce the next JSON action."
        )
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, asst_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await llm.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            action_text = response.choices[0].message.content
            action = json.loads(action_text)
            action_str = json.dumps(action).replace('"', "'")
        except Exception as e:
            error_str = str(e)
            action_str = "null"
            done = True
            log_step(step=step, reward=0.0)
            break
            
        history.append((prompt, action_text))
        
        try:
            result = await client.step(action)
            obs = result["observation"]
            reward = result["reward"]["value"]
            done = result["done"]
            info = result["info"]
            error = None
        except Exception as e:
            reward = 0.0
            done = True
            error = "HTTP error"
            info = {}
            
        rewards.append(reward)
        steps_taken = step
        
        log_step(step=step, reward=reward)
        
        if done:
            score = info.get("final_score", 0.0)
            break
            
    log_end(task=task_id, score=score, steps=steps_taken)

async def main():
    api_key = os.getenv("OPENAI_API_KEY", "dummy_key_for_testing")

        
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    client = EnvClient(base_url=env_url)
    
    for task_id in ["easy", "medium", "hard"]:
        await run_task(task_id, client, llm, model_name)

if __name__ == "__main__":
    asyncio.run(main())
