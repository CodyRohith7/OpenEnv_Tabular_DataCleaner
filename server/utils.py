import pandas as pd
from typing import List, Optional


def _normalize_str(v) -> str:
    """Stringify and normalize a cell value for comparison."""
    s = str(v).strip()
    # Collapse float representations: 1000.0 → 1000
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    if s.lower() in ("nan", "none", ""):
        return "nan"
    return s


def compute_cell_accuracy(
    df_current: pd.DataFrame,
    df_gold: pd.DataFrame,
    columns: List[str],
) -> float:
    """
    Order-invariant cell accuracy: compares multiset distributions of values
    in each acc_col between current and gold DataFrames.

    Numeric tolerance: values that differ by < 0.01 are treated as equal
    (handles float rounding from currency conversion).
    """
    if df_current.empty or df_gold.empty or not columns:
        return 0.0

    total_cells = len(columns) * len(df_gold)
    matches = 0

    for col in columns:
        if col not in df_current.columns or col not in df_gold.columns:
            continue

        cur_vals = [_normalize_str(v) for v in df_current[col]]
        gold_vals = [_normalize_str(v) for v in df_gold[col]]

        # Build multiset counts for gold
        gold_counts: dict = {}
        for v in gold_vals:
            gold_counts[v] = gold_counts.get(v, 0) + 1

        # Build multiset counts for current
        cur_counts: dict = {}
        for v in cur_vals:
            cur_counts[v] = cur_counts.get(v, 0) + 1

        # Exact matches first
        col_matches = 0
        for k, cnt in gold_counts.items():
            col_matches += min(cnt, cur_counts.get(k, 0))

        # Numeric near-match fallback for un-matched gold items
        try:
            unmatched_gold: dict = {}
            for k, cnt in gold_counts.items():
                used = min(cnt, cur_counts.get(k, 0))
                remaining = cnt - used
                if remaining > 0:
                    unmatched_gold[k] = remaining

            unmatched_cur: dict = {}
            for k, cnt in cur_counts.items():
                used = min(cnt, gold_counts.get(k, 0))
                remaining = cnt - used
                if remaining > 0:
                    unmatched_cur[k] = remaining

            gold_floats = []
            for k, cnt in unmatched_gold.items():
                try:
                    gold_floats.extend([float(k)] * cnt)
                except ValueError:
                    pass

            for k, cnt in unmatched_cur.items():
                try:
                    fv = float(k)
                    for i, gf in enumerate(gold_floats):
                        if abs(fv - gf) < 0.01:
                            col_matches += 1
                            gold_floats.pop(i)
                            cnt -= 1
                            break
                except ValueError:
                    pass
        except Exception:
            pass

        matches += col_matches

    return round(matches / max(1, total_cells), 6)


def compute_dedup_f1(
    df_current: pd.DataFrame,
    df_gold: pd.DataFrame,
    keys: List[str],
) -> float:
    """
    F1 score on the set of unique key-tuples present in current vs gold.
    Empty keys → return 1.0 (dedup not applicable).
    """
    if not keys:
        return 1.0
    try:
        valid_cur = [k for k in keys if k in df_current.columns]
        valid_gold = [k for k in keys if k in df_gold.columns]
        if not valid_cur or not valid_gold:
            return 0.0

        cur_keys = set(
            tuple(_normalize_str(x) for x in row)
            for row in df_current[valid_cur].values
        )
        gold_keys = set(
            tuple(_normalize_str(x) for x in row)
            for row in df_gold[valid_gold].values
        )
    except Exception:
        return 0.0

    tp = len(cur_keys & gold_keys)
    fp = len(cur_keys - gold_keys)
    fn = len(gold_keys - cur_keys)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return round(2 * precision * recall / (precision + recall), 6)


def compute_schema_score(current_cols: List[str], target_cols: Optional[List[str]]) -> float:
    """
    Score the current schema against the target:
      1.0 — exact column list and order match
      0.8 — same columns, wrong order
      0–0.5 — partial: ratio of target columns present
    """
    if target_cols is None:
        return 1.0
    if current_cols == target_cols:
        return 1.0
    if set(current_cols) == set(target_cols):
        return 0.8
    present = sum(1 for c in target_cols if c in current_cols)
    return round(present / max(1, len(target_cols)) * 0.5, 6)
