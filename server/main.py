from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .environment import DataCleaningEnv, TASKS
from .models import Action, Observation, OpType, Reward, State

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv: Enterprise Data Cleaning Environment",
    description=(
        "A production-grade reinforcement learning environment for tabular "
        "data engineering tasks — from basic CRM normalization (easy) to "
        "full ETL pipelines with JSON parsing, PII redaction, and outlier "
        "removal (extreme). Built for the Meta PyTorch Hackathon 2026."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global environment instance
# ─────────────────────────────────────────────────────────────────────────────

env = DataCleaningEnv()

# ─────────────────────────────────────────────────────────────────────────────
# Load dashboard HTML once at startup
# ─────────────────────────────────────────────────────────────────────────────

_DASH_PATH = Path(__file__).parent / "static" / "index.html"
_DASHBOARD_HTML: str = (
    _DASH_PATH.read_text(encoding="utf-8")
    if _DASH_PATH.exists()
    else "<h1>Dashboard unavailable — static/index.html not found.</h1>"
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    difficulty: str
    description: str
    max_steps: int
    available_ops: List[str]


class GoldPreview(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    """Interactive data-cleaning playground dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/health", tags=["System"])
def health():
    """Health probe used by Dockerfile HEALTHCHECK and HF Space readiness."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "tasks": list(TASKS.keys()),
    }


@app.get("/tasks", response_model=List[TaskInfo], tags=["Environment"])
def list_tasks():
    """List all available tasks with difficulty and metadata."""
    ops = [op.value for op in OpType]
    return [
        TaskInfo(
            id=tid,
            difficulty=info["difficulty"],
            description=info["instructions"].split("\n")[0],
            max_steps=info["max_steps"],
            available_ops=ops,
        )
        for tid, info in TASKS.items()
    ]


@app.get("/tasks/{task_id}/gold-preview", response_model=GoldPreview, tags=["Environment"])
def gold_preview(task_id: str):
    """Return the first 10 rows of the gold dataset for a task (for UI display)."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    import pandas as pd
    gold_file = TASKS[task_id]["gold_file"]
    if not Path(gold_file).exists():
        raise HTTPException(status_code=404, detail="Gold file not found.")
    df = pd.read_csv(gold_file, nrows=10)
    return GoldPreview(
        columns=list(df.columns),
        rows=df.fillna("NaN").astype(str).to_dict(orient="records"),
    )


@app.post("/reset", response_model=Observation, tags=["Environment"])
def reset_env(req: ResetRequest = None):
    """
    Reset the environment to a fresh state for the given task.
    Returns the initial Observation.
    """
    if req is None:
        req = ResetRequest(task_id="easy")
    return env.reset(req.task_id)


@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step_env(action: Action):
    """
    Apply one Action and receive the next Observation, Reward, done flag, and info.
    Must call /reset first.
    """
    if env.episode_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised — call POST /reset first.",
        )
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=State, tags=["Environment"])
def get_state():
    """Return the lightweight current state (episode_id, task, step count, score)."""
    if env.episode_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised — call POST /reset first.",
        )
    return env.state()
