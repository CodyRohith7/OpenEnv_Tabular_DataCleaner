import pandas as pd

def compute_cell_accuracy(df_current: pd.DataFrame, df_gold: pd.DataFrame, columns: list) -> float:
    """Computes an order-invariant cell accuracy metric."""
    if len(df_current) == 0 or len(df_gold) == 0 or not columns:
        return 0.0
        
    total_cells = len(columns) * len(df_gold)
    matches = 0
    
    for col in columns:
        if col in df_current.columns and col in df_gold.columns:
            # Normalize strings for comparison: handle numeric decimals (.0)
            def normalize(v):
                s = str(v).strip()
                if s.endswith('.0'): s = s[:-2]
                if s == 'nan' or s == 'None': return 'nan'
                return s

            cur_vals = [normalize(v) for v in df_current[col]]
            gold_vals = [normalize(v) for v in df_gold[col]]
            
            cur_counts = {}
            for v in cur_vals: cur_counts[v] = cur_counts.get(v, 0) + 1
            
            gold_counts = {}
            for v in gold_vals: gold_counts[v] = gold_counts.get(v, 0) + 1
            
            col_matches = 0
            for k, v in gold_counts.items():
                col_matches += min(v, cur_counts.get(k, 0))
            matches += col_matches
            
    return matches / max(1, total_cells)

def compute_dedup_f1(df_current: pd.DataFrame, df_gold: pd.DataFrame, keys: list) -> float:
    """Computes F1 score on the availability of unique dimension keys."""
    if not keys:
        return 1.0 
    try:
        cur_keys = set(tuple(x) for x in df_current[keys].values)
        gold_keys = set(tuple(x) for x in df_gold[keys].values)
    except Exception:
        return 0.0
    
    tp = len(cur_keys.intersection(gold_keys))
    fp = len(cur_keys - gold_keys)
    fn = len(gold_keys - cur_keys)
    
    if tp == 0: return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)
