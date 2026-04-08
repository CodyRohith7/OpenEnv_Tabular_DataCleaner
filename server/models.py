from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class OpType(str, Enum):
    FILL_MISSING = "FILL_MISSING"
    NORMALIZE_CASE = "NORMALIZE_CASE"
    STRIP_WHITESPACE = "STRIP_WHITESPACE"
    DEDUP_ROWS = "DEDUP_ROWS"
    PARSE_DATE = "PARSE_DATE"
    RENAME_COLUMN = "RENAME_COLUMN"
    DROP_COLUMN = "DROP_COLUMN"
    EXTRACT_MONTH = "EXTRACT_MONTH"
    CONVERT_CURRENCY = "CONVERT_CURRENCY"
    GROUPBY_SUM = "GROUPBY_SUM"

class Action(BaseModel):
    op: OpType = Field(..., description="The cleaning operation to apply.")
    column: Optional[str] = Field(None, description="Target column name where relevant. Groupby uses comma separation.")
    value: Optional[str] = Field(None, description="Literal value, replacement, or new column name.")
    pattern: Optional[str] = Field(None, description="Pattern or format string, e.g., date format.")

class RowPreview(BaseModel):
    values: Dict[str, Any]

class Observation(BaseModel):
    task_id: str = Field(..., description="Task name: easy/medium/hard.")
    instructions: str = Field(..., description="Natural language task description.")
    preview_original: List[RowPreview] = Field(..., description="Sample of original data.")
    preview_current: List[RowPreview] = Field(..., description="Sample of current transformed data.")
    cell_accuracy: float = Field(..., ge=0.0, le=1.0)
    duplicate_rate: float = Field(..., ge=0.0, le=1.0)
    missing_rate: float = Field(..., ge=0.0, le=1.0)
    step_index: int = Field(..., ge=0)

class State(BaseModel):
    episode_id: str
    task_id: str
    step_count: int

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward in [0,1] for this step.")
    delta_accuracy: float = Field(..., description="Change in cell accuracy vs previous step.")
    penalties: float = Field(..., description="Negative penalties this step.")
