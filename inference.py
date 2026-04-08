import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from typing import List

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = error if error else "null"
    # Action without quotes, reward 2 decimals, done lower, error null or string
    print(f'[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_str}', flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
    obs = await client.reset(task_id)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="TabularDataCleaning", model=model_name)
    
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
            log_step(step=step, action=action_str, reward=0.0, done=True, error=error_str)
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
        
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            score = info.get("final_score", 0.0)
            break
            
    success = score >= 0.8
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return
        
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    client = EnvClient(base_url=env_url)
    
    for task_id in ["easy", "medium", "hard"]:
        await run_task(task_id, client, llm, model_name)

if __name__ == "__main__":
    asyncio.run(main())
