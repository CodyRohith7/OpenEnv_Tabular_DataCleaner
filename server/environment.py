import uuid
import os
import pandas as pd
from typing import Tuple, Dict, Any
from .models import Action, Observation, State, Reward, RowPreview, OpType
from .utils import compute_cell_accuracy, compute_dedup_f1

TASKS = {
    "easy": {
        "instructions": "Clean up the 'city' column: strip extra spaces, normalize to Title Case, and replace missing/NaN values with 'Unknown'.",
        "raw_file": "server/datasets/task_easy_raw.csv",
        "gold_file": "server/datasets/task_easy_gold.csv",
        "max_steps": 5,
        "acc_cols": ["city"],
        "dedup_keys": []
    },
    "medium": {
        "instructions": "Standardize 'order_date' to YYYY-MM-DD format (use PARSE_DATE pattern='%Y/%m/%d' etc). Normalize 'country' to Title Case. Deduplicate rows entirely so there are no exact duplicates.",
        "raw_file": "server/datasets/task_medium_raw.csv",
        "gold_file": "server/datasets/task_medium_gold.csv",
        "max_steps": 10,
        "acc_cols": ["order_date", "country"],
        "dedup_keys": ["order_id", "line_item"]
    },
    "hard": {
        "instructions": "1) EXTRACT_MONTH from 'timestamp' column into 'month'. 2) CONVERT_CURRENCY targeting 'currency' to normalize amounts natively. 3) GROUPBY_SUM on column='month,country' taking value='amount'. 4) RENAME_COLUMN 'amount' to 'total_revenue'.",
        "raw_file": "server/datasets/task_hard_raw.csv",
        "gold_file": "server/datasets/task_hard_gold.csv",
        "max_steps": 15,
        "acc_cols": ["month", "country", "total_revenue"],
        "dedup_keys": ["month", "country"]
    }
}

class DataCleaningEnv:
    def __init__(self):
        self.episode_id = None
        self.task_id = "easy"
        self.step_count = 0
        self.df = pd.DataFrame()
        self.df_initial = pd.DataFrame()
        self.df_gold = pd.DataFrame()
        self.task_info = None
        self.prev_metric = 0.0

    def load_data(self, task_id: str):
        self.task_info = TASKS[task_id]
        if os.path.exists(self.task_info["raw_file"]):
            self.df = pd.read_csv(self.task_info["raw_file"])
            self.df_initial = self.df.copy()
            self.df_gold = pd.read_csv(self.task_info["gold_file"])

    def get_preview(self, df: pd.DataFrame, n=3) -> list:
        if df is None or len(df) == 0:
            return []
        records = df.head(n).astype(str).to_dict(orient="records")
        return [RowPreview(values=r) for r in records]

    def _compute_composite_metric(self) -> float:
        if self.task_id == "easy":
            return compute_cell_accuracy(self.df, self.df_gold, self.task_info["acc_cols"])
        elif self.task_id == "medium":
            acc = compute_cell_accuracy(self.df, self.df_gold, self.task_info["acc_cols"])
            dedup = compute_dedup_f1(self.df, self.df_gold, self.task_info["dedup_keys"])
            return 0.6 * acc + 0.4 * dedup
        elif self.task_id == "hard":
            acc = compute_cell_accuracy(self.df, self.df_gold, self.task_info["acc_cols"])
            dedup = compute_dedup_f1(self.df, self.df_gold, self.task_info["dedup_keys"])
            
            schema_score = 0.0
            if list(self.df.columns) == ["month", "country", "total_revenue"]:
                schema_score = 1.0
            elif set(self.df.columns) == {"month", "country", "total_revenue"}:
                schema_score = 0.8
                
            return 0.3 * schema_score + 0.3 * dedup + 0.4 * acc
        return 0.0

    def apply_action(self, action: Action) -> Tuple[float, float]:
        penalties = 0.0
        try:
            col = action.column
            if action.op == OpType.FILL_MISSING:
                if col and col in self.df.columns:
                    val = action.value or "Unknown"
                    self.df[col] = self.df[col].fillna(val)
                else: penalties += 0.1
            elif action.op == OpType.NORMALIZE_CASE:
                if col and col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: str(x).title() if pd.notnull(x) else x)
                else: penalties += 0.1
            elif action.op == OpType.STRIP_WHITESPACE:
                if col and col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
                else: penalties += 0.1
            elif action.op == OpType.DEDUP_ROWS:
                if col:
                    subset = [c.strip() for c in col.split(",") if c.strip() in self.df.columns]
                    self.df.drop_duplicates(subset=subset, inplace=True)
                else:
                    self.df.drop_duplicates(inplace=True)
            elif action.op == OpType.PARSE_DATE:
                if col and col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                else: penalties += 0.1
            elif action.op == OpType.RENAME_COLUMN:
                if col and col in self.df.columns and action.value:
                    self.df.rename(columns={col: action.value}, inplace=True)
                else: penalties += 0.1
            elif action.op == OpType.DROP_COLUMN:
                if col and col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
                else: penalties += 0.1
            elif action.op == OpType.EXTRACT_MONTH:
                if col and action.value and col in self.df.columns:
                    self.df[action.value] = pd.to_datetime(self.df[col], errors='coerce').dt.strftime('%Y-%m')
                else: penalties += 0.1
            elif action.op == OpType.CONVERT_CURRENCY:
                if col and col in self.df.columns and 'amount' in self.df.columns:
                    mask = self.df[col] == 'EUR'
                    self.df.loc[mask, 'amount'] = pd.to_numeric(self.df.loc[mask, 'amount'], errors='coerce') * 1.1
                    self.df[col] = 'USD'
                else: penalties += 0.1
            elif action.op == OpType.GROUPBY_SUM:
                if col and action.value:
                    # Clean up grouping columns: split by comma, strip whitespace, and verify existence
                    cols_to_group = [c.strip() for c in col.split(",") if c.strip() in self.df.columns]
                    val_col = action.value.strip()
                    
                    if cols_to_group and val_col in self.df.columns:
                        # Ensure value column is numeric before grouping
                        self.df[val_col] = pd.to_numeric(self.df[val_col], errors='coerce').fillna(0)
                        # Perform grouping and reset index to keep it as a flat dataframe
                        self.df = self.df.groupby(cols_to_group, as_index=False)[val_col].sum()
                        # Round to avoid float precision issues in metric
                        self.df[val_col] = self.df[val_col].round(2)
                    else:
                        penalties += 0.1
                else:
                    penalties += 0.1
        except Exception:
            penalties += 0.2
            
        curr_metric = self._compute_composite_metric()
        delta = curr_metric - self.prev_metric
        self.prev_metric = curr_metric
        return delta, penalties

    def generate_observation(self) -> Observation:
        metric = self._compute_composite_metric()
        missing = self.df.isna().sum().sum() / max(1, self.df.size)
        dup = self.df.duplicated().sum() / max(1, len(self.df))
        
        return Observation(
            task_id=self.task_id,
            instructions=self.task_info["instructions"],
            preview_original=self.get_preview(self.df_initial),
            preview_current=self.get_preview(self.df),
            cell_accuracy=float(metric),
            duplicate_rate=float(dup),
            missing_rate=float(missing),
            step_index=self.step_count
        )

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            task_id = "easy"
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.step_count = 0
        self.load_data(task_id)
        self.prev_metric = self._compute_composite_metric()
        return self.generate_observation()

    def state(self) -> State:
        return State(
            episode_id=self.episode_id or "",
            task_id=self.task_id,
            step_count=self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        delta, penalties = self.apply_action(action)
        
        obs = self.generate_observation()
        
        raw_reward = 0.5 + (delta * 0.5) - penalties
        reward_val = max(0.0, min(1.0, raw_reward))
        
        r = Reward(value=reward_val, delta_accuracy=delta, penalties=penalties)
        
        max_steps = self.task_info["max_steps"]
        composite = self._compute_composite_metric()
        
        done = False
        if self.step_count >= max_steps or composite >= 0.999:
            done = True
            
        info = {}
        if done:
            raw_score = composite + 0.05 * (max_steps - self.step_count) / max_steps
            final_score = min(max(raw_score, 0.001), 0.999)
            info["final_score"] = float(final_score)
            
        return obs, r, done, info
