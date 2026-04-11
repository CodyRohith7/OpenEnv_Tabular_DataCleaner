from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OpType(str, Enum):
    """All supported data cleaning operations."""
    # Basic cleaning
    FILL_MISSING = "FILL_MISSING"
    NORMALIZE_CASE = "NORMALIZE_CASE"
    STRIP_WHITESPACE = "STRIP_WHITESPACE"
    CAST_NUMERIC = "CAST_NUMERIC"
    # Structural
    DEDUP_ROWS = "DEDUP_ROWS"
    DROP_COLUMN = "DROP_COLUMN"
    RENAME_COLUMN = "RENAME_COLUMN"
    # Temporal
    PARSE_DATE = "PARSE_DATE"
    EXTRACT_MONTH = "EXTRACT_MONTH"
    # Financial
    CONVERT_CURRENCY = "CONVERT_CURRENCY"
    # Aggregation
    GROUPBY_SUM = "GROUPBY_SUM"
    # Advanced / Extreme tier
    PII_REDACT = "PII_REDACT"
    EXTRACT_JSON = "EXTRACT_JSON"
    DROP_OUTLIERS = "DROP_OUTLIERS"


class Action(BaseModel):
    """A declarative data-engineering operation for the agent to apply."""
    op: OpType = Field(
        ...,
        description="The cleaning operation to apply.",
    )
    column: Optional[str] = Field(
        None,
        description=(
            "Target column. For multi-column ops (DEDUP_ROWS, GROUPBY_SUM) "
            "use comma-separated string, e.g. 'month,country'."
        ),
    )
    value: Optional[str] = Field(
        None,
        description=(
            "Replacement value, new column name, JSON key to extract, "
            "or value column for GROUPBY_SUM."
        ),
    )
    pattern: Optional[str] = Field(
        None,
        description=(
            "Date format string for PARSE_DATE, or target column name "
            "for EXTRACT_JSON output."
        ),
    )


class RowPreview(BaseModel):
    values: Dict[str, Any]


class Observation(BaseModel):
    """Full observable state returned after each reset() or step()."""
    task_id: str = Field(..., description="Task identifier: easy / medium / hard / extreme.")
    instructions: str = Field(..., description="Natural language description of the ETL task.")
    # Data previews
    preview_original: List[RowPreview] = Field(..., description="First 5 rows of the raw dataset.")
    preview_current: List[RowPreview] = Field(..., description="First 5 rows of the current dataset.")
    # Schema info — critical for LLM agents to form correct actions
    columns: List[str] = Field(..., description="Current column names.")
    column_types: Dict[str, str] = Field(..., description="Dtype of each column, e.g. {'amount': 'float64'}.")
    row_count: int = Field(..., description="Current number of rows.")
    # Metrics
    cell_accuracy: float = Field(..., ge=0.0, le=1.0, description="Composite score vs gold dataset [0, 1].")
    schema_score: float = Field(..., ge=0.0, le=1.0, description="Schema match against target structure [0, 1].")
    duplicate_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of duplicate rows.")
    missing_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of NaN cells.")
    step_index: int = Field(..., ge=0, description="Steps taken so far this episode.")
    # Self-documentation for agents
    available_ops: List[str] = Field(..., description="All valid OpType values the agent may use.")


class State(BaseModel):
    """Lightweight state snapshot for polling / bookkeeping."""
    episode_id: str = Field(..., description="UUID of the current episode.")
    task_id: str = Field(..., description="Active task identifier.")
    step_count: int = Field(..., description="Steps taken so far.")
    composite_score: float = Field(..., description="Current composite metric vs gold.")


class Reward(BaseModel):
    """Structured reward signal returned by step()."""
    value: float = Field(..., ge=0.0, le=1.0, description="Step reward [0, 1].")
    delta_accuracy: float = Field(..., description="Change in composite score this step (positive = improvement).")
    penalties: float = Field(..., description="Penalty incurred for invalid/destructive actions.")
