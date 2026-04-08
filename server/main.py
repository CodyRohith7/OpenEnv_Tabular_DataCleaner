from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from .environment import DataCleaningEnv
from .models import Action, Observation, State, Reward

app = FastAPI(title="Tabular Data Cleaning Environment")
env = DataCleaningEnv()

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the OpenEnv Tabular Data Cleaning Environment",
        "spec": "OpenEnv 1.0",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = None):
    # Handle empty POST requests gracefully (required by some graders)
    if req is None:
        req = ResetRequest(task_id="easy")
    return env.reset(req.task_id)

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    if env.episode_id is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=State)
def get_state():
    if env.episode_id is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return env.state()
