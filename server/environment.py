import uuid
import os
import re
import json
import pandas as pd
from typing import Tuple, Dict, Any

from .models import Action, Observation, State, Reward, RowPreview, OpType
from .utils import compute_cell_accuracy, compute_dedup_f1, compute_schema_score

# ─────────────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict] = {
    "easy": {
        "difficulty": "Easy",
        "instructions": (
            "You are cleaning a CRM contact export. Apply these operations in order:\n"
            "1) STRIP_WHITESPACE on 'full_name' — remove leading/trailing spaces.\n"
            "2) STRIP_WHITESPACE on 'region' — remove leading/trailing spaces.\n"
            "3) NORMALIZE_CASE on 'full_name' — convert to Title Case.\n"
            "4) NORMALIZE_CASE on 'region' — convert to Title Case.\n"
            "5) FILL_MISSING on 'region' with value='Unknown' — impute blank regions."
        ),
        "raw_file": "server/datasets/task_easy_raw.csv",
        "gold_file": "server/datasets/task_easy_gold.csv",
        "max_steps": 8,
        "acc_cols": ["full_name", "region"],
        "dedup_keys": [],
        "target_schema": None,
    },
    "medium": {
        "difficulty": "Medium",
        "instructions": (
            "You are standardizing a sales order export. Apply in order:\n"
            "1) STRIP_WHITESPACE on 'country' — remove padding whitespace.\n"
            "2) NORMALIZE_CASE on 'country' — convert to Title Case.\n"
            "3) PARSE_DATE on 'order_date' — normalize all formats to YYYY-MM-DD.\n"
            "4) DEDUP_ROWS with column='order_id,line_item' — remove exact duplicate records."
        ),
        "raw_file": "server/datasets/task_medium_raw.csv",
        "gold_file": "server/datasets/task_medium_gold.csv",
        "max_steps": 10,
        "acc_cols": ["order_date", "country"],
        "dedup_keys": ["order_id", "line_item"],
        "target_schema": None,
    },
    "hard": {
        "difficulty": "Hard",
        "instructions": (
            "Transform a multi-currency revenue ledger into a monthly analytics table:\n"
            "1) EXTRACT_MONTH: column='timestamp', value='month' — extract YYYY-MM period.\n"
            "2) CONVERT_CURRENCY: column='currency' — normalize EUR→USD (×1.08), GBP→USD (×1.27).\n"
            "3) GROUPBY_SUM: column='month,country', value='amount' — aggregate total revenue.\n"
            "4) RENAME_COLUMN: column='amount', value='total_revenue' — rename the sum column.\n"
            "Target schema: [month, country, total_revenue]."
        ),
        "raw_file": "server/datasets/task_hard_raw.csv",
        "gold_file": "server/datasets/task_hard_gold.csv",
        "max_steps": 15,
        "acc_cols": ["month", "country", "total_revenue"],
        "dedup_keys": ["month", "country"],
        "target_schema": ["month", "country", "total_revenue"],
    },
    "extreme": {
        "difficulty": "Extreme",
        "instructions": (
            "Process a raw marketing lead export through a full ETL pipeline:\n"
            "1) EXTRACT_JSON: column='metadata_json', value='source', pattern='lead_source' "
            "— parse JSON strings and extract the 'source' key into a new column.\n"
            "2) CAST_NUMERIC: column='revenue_usd' — convert string values to float.\n"
            "3) PII_REDACT: column='contact_email' — mask email addresses with [REDACTED].\n"
            "4) DROP_OUTLIERS: column='revenue_usd' — remove rows where z-score > 3.\n"
            "5) NORMALIZE_CASE: column='region' — convert to Title Case.\n"
            "6) GROUPBY_SUM: column='lead_source,region', value='revenue_usd' — aggregate.\n"
            "7) RENAME_COLUMN: column='revenue_usd', value='total_revenue' — finalize schema.\n"
            "Target schema: [lead_source, region, total_revenue]."
        ),
        "raw_file": "server/datasets/task_extreme_raw.csv",
        "gold_file": "server/datasets/task_extreme_gold.csv",
        "max_steps": 20,
        "acc_cols": ["lead_source", "region", "total_revenue"],
        "dedup_keys": ["lead_source", "region"],
        "target_schema": ["lead_source", "region", "total_revenue"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaningEnv:
    def __init__(self):
        self.episode_id: str | None = None
        self.task_id: str = "easy"
        self.step_count: int = 0
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_initial: pd.DataFrame = pd.DataFrame()
        self.df_gold: pd.DataFrame = pd.DataFrame()
        self.task_info: Dict | None = None
        self.prev_metric: float = 0.0

    # ── Data Loading ────────────────────────────────────────────────────────

    def load_data(self, task_id: str) -> None:
        self.task_info = TASKS[task_id]
        raw = self.task_info["raw_file"]
        gold = self.task_info["gold_file"]
        if os.path.exists(raw):
            self.df = pd.read_csv(raw, dtype=str, keep_default_na=True)
            self.df_initial = self.df.copy()
        if os.path.exists(gold):
            self.df_gold = pd.read_csv(gold, dtype=str, keep_default_na=True)

    def get_preview(self, df: pd.DataFrame, n: int = 5) -> list:
        if df is None or df.empty:
            return []
        records = df.head(n).fillna("NaN").astype(str).to_dict(orient="records")
        return [RowPreview(values=r) for r in records]

    # ── Metrics ─────────────────────────────────────────────────────────────

    def _schema_score(self) -> float:
        target = self.task_info.get("target_schema")
        return compute_schema_score(list(self.df.columns), target)

    def _composite(self) -> float:
        acc = compute_cell_accuracy(
            self.df, self.df_gold, self.task_info["acc_cols"]
        )
        if self.task_id == "easy":
            return acc
        dedup = compute_dedup_f1(
            self.df, self.df_gold, self.task_info["dedup_keys"]
        )
        if self.task_id == "medium":
            return round(0.6 * acc + 0.4 * dedup, 6)
        # hard / extreme: acc + dedup + schema
        schema = self._schema_score()
        return round(0.4 * acc + 0.3 * dedup + 0.3 * schema, 6)

    # ── Action Execution ────────────────────────────────────────────────────

    def apply_action(self, action: Action) -> Tuple[float, float]:  # (delta, penalty)
        penalties: float = 0.0
        col = action.column

        try:
            op = action.op

            if op == OpType.FILL_MISSING:
                if col and col in self.df.columns:
                    val = action.value or "Unknown"
                    self.df[col] = (
                        self.df[col]
                        .fillna(val)
                        .replace({"NaN": val, "nan": val, "": val})
                    )
                else:
                    penalties += 0.1

            elif op == OpType.NORMALIZE_CASE:
                if col and col in self.df.columns:
                    self.df[col] = self.df[col].apply(
                        lambda x: str(x).title()
                        if pd.notnull(x) and str(x).lower() not in ("nan", "")
                        else x
                    )
                else:
                    penalties += 0.1

            elif op == OpType.STRIP_WHITESPACE:
                if col and col in self.df.columns:
                    self.df[col] = self.df[col].apply(
                        lambda x: str(x).strip()
                        if pd.notnull(x) and str(x).lower() != "nan"
                        else x
                    )
                else:
                    penalties += 0.1

            elif op == OpType.CAST_NUMERIC:
                if col and col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                else:
                    penalties += 0.1

            elif op == OpType.DEDUP_ROWS:
                if col:
                    subset = [
                        c.strip()
                        for c in col.split(",")
                        if c.strip() in self.df.columns
                    ]
                    if subset:
                        self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
                    else:
                        penalties += 0.1
                else:
                    self.df = self.df.drop_duplicates().reset_index(drop=True)

            elif op == OpType.PARSE_DATE:
                if col and col in self.df.columns:
                    # First pass: pandas auto-inference (handles ISO, YYYY/MM/DD, etc.)
                    try:
                        parsed = pd.to_datetime(
                            self.df[col], errors="coerce", dayfirst=False, format="mixed"
                        )
                    except TypeError:
                        parsed = pd.to_datetime(
                            self.df[col], errors="coerce", dayfirst=False,
                            infer_datetime_format=True,
                        )
                    # Second pass: for remaining NaTs, try explicit MM/DD/YYYY
                    nat_mask = parsed.isna()
                    if nat_mask.any():
                        fallback = pd.to_datetime(
                            self.df.loc[nat_mask, col], format="%m/%d/%Y", errors="coerce"
                        )
                        parsed = parsed.where(~nat_mask, fallback)
                    self.df[col] = parsed.dt.strftime("%Y-%m-%d")
                else:
                    penalties += 0.1

            elif op == OpType.RENAME_COLUMN:
                if col and col in self.df.columns and action.value:
                    self.df.rename(columns={col: action.value}, inplace=True)
                else:
                    penalties += 0.1

            elif op == OpType.DROP_COLUMN:
                if col and col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
                else:
                    penalties += 0.1

            elif op == OpType.EXTRACT_MONTH:
                if col and action.value and col in self.df.columns:
                    self.df[action.value] = (
                        pd.to_datetime(self.df[col], errors="coerce")
                        .dt.strftime("%Y-%m")
                    )
                else:
                    penalties += 0.1

            elif op == OpType.CONVERT_CURRENCY:
                if col and col in self.df.columns:
                    amount_col = action.value or "amount"
                    if amount_col in self.df.columns:
                        # Cast to float64 first to avoid FutureWarning on in-place multiply
                        self.df[amount_col] = pd.to_numeric(
                            self.df[amount_col], errors="coerce"
                        ).astype("float64")
                        rate_map = {"EUR": 1.08, "GBP": 1.27}
                        for currency, rate in rate_map.items():
                            mask = self.df[col].str.strip() == currency
                            if mask.any():
                                vals = self.df.loc[mask, amount_col].astype("float64")
                                self.df[amount_col] = self.df[amount_col].astype("float64")
                                self.df.loc[mask, amount_col] = (vals * rate).round(2)
                                self.df.loc[mask, col] = "USD"
                    else:
                        penalties += 0.1
                else:
                    penalties += 0.1

            elif op == OpType.GROUPBY_SUM:
                if col and action.value:
                    group_cols = [
                        c.strip()
                        for c in col.split(",")
                        if c.strip() in self.df.columns
                    ]
                    val_col = action.value.strip()
                    if group_cols and val_col in self.df.columns:
                        self.df[val_col] = pd.to_numeric(
                            self.df[val_col], errors="coerce"
                        ).fillna(0)
                        self.df = (
                            self.df.groupby(group_cols, as_index=False)[val_col]
                            .sum()
                        )
                        self.df[val_col] = self.df[val_col].round(2)
                    else:
                        penalties += 0.1
                else:
                    penalties += 0.1

            # ── Advanced (Extreme tier) ─────────────────────────────────────

            elif op == OpType.PII_REDACT:
                if col and col in self.df.columns:
                    email_re = re.compile(
                        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
                    )
                    self.df[col] = self.df[col].apply(
                        lambda x: email_re.sub("[REDACTED]", str(x))
                        if pd.notnull(x)
                        else x
                    )
                else:
                    penalties += 0.1

            elif op == OpType.EXTRACT_JSON:
                if (
                    col
                    and col in self.df.columns
                    and action.value
                    and action.pattern
                ):
                    key_name = action.value
                    target_col = action.pattern

                    def _extract(val):
                        try:
                            return json.loads(str(val)).get(key_name)
                        except Exception:
                            return None

                    self.df[target_col] = self.df[col].apply(_extract)
                else:
                    penalties += 0.1

            elif op == OpType.DROP_OUTLIERS:
                if col and col in self.df.columns:
                    numeric_col = pd.to_numeric(self.df[col], errors="coerce")
                    valid = numeric_col.dropna()
                    if len(valid) > 1:
                        mean = valid.mean()
                        std = valid.std()
                        if std > 0:
                            z = (numeric_col - mean).abs() / std
                            self.df = self.df[z <= 3.0].reset_index(drop=True)
                else:
                    penalties += 0.1

        except Exception:
            penalties += 0.2

        curr = self._composite()
        delta = round(curr - self.prev_metric, 6)
        self.prev_metric = curr
        return delta, penalties

    # ── Observation ─────────────────────────────────────────────────────────

    def generate_observation(self) -> Observation:
        composite = self._composite()
        schema = self._schema_score()
        missing = self.df.isna().sum().sum() / max(1, self.df.size)
        dup = self.df.duplicated().sum() / max(1, len(self.df))
        col_types = {c: str(self.df[c].dtype) for c in self.df.columns}

        return Observation(
            task_id=self.task_id,
            instructions=self.task_info["instructions"],
            preview_original=self.get_preview(self.df_initial),
            preview_current=self.get_preview(self.df),
            columns=list(self.df.columns),
            column_types=col_types,
            row_count=len(self.df),
            cell_accuracy=float(composite),
            schema_score=float(schema),
            duplicate_rate=float(dup),
            missing_rate=float(missing),
            step_index=self.step_count,
            available_ops=[op.value for op in OpType],
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            task_id = "easy"
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.step_count = 0
        self.load_data(task_id)
        self.prev_metric = self._composite()
        return self.generate_observation()

    def state(self) -> State:
        return State(
            episode_id=self.episode_id or "",
            task_id=self.task_id,
            step_count=self.step_count,
            composite_score=float(self._composite()),
        )

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        delta, penalties = self.apply_action(action)
        obs = self.generate_observation()

        # ── Reward shaping: meaningful partial progress signals ───────────
        if penalties > 0:
            # Invalid / destructive action — clear negative signal
            reward_val = max(0.0, 0.1 - penalties)
        elif delta > 0.0:
            # Genuine improvement — proportional reward with a floor
            reward_val = min(1.0, 0.3 + delta * 5.0)
        elif delta == 0.0:
            # Valid op but no change — tiny positive to encourage exploration
            reward_val = 0.05
        else:
            # Made things worse
            reward_val = 0.0

        r = Reward(value=round(reward_val, 4), delta_accuracy=delta, penalties=penalties)

        # ── Episode termination ──────────────────────────────────────────
        max_steps = self.task_info["max_steps"]
        composite = self._composite()
        done = self.step_count >= max_steps or composite >= 0.999

        info: Dict[str, Any] = {}
        if done:
            steps_remaining = max(0, max_steps - self.step_count)
            efficiency_bonus = round(0.05 * steps_remaining / max_steps, 4)
            final_score = round(min(max(composite + efficiency_bonus, 0.001), 0.999), 4)
            info = {
                "final_score": final_score,
                "composite_metric": float(composite),
                "steps_used": self.step_count,
                "efficiency_bonus": efficiency_bonus,
            }

        return obs, r, done, info
