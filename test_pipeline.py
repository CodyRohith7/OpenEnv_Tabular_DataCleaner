import sys
sys.path.insert(0, '.')
from server.environment import DataCleaningEnv
from server.models import Action

env = DataCleaningEnv()

# EASY
print('=== EASY ===')
env.reset('easy')
for a in [
    {'op':'STRIP_WHITESPACE','column':'full_name'},
    {'op':'STRIP_WHITESPACE','column':'region'},
    {'op':'NORMALIZE_CASE','column':'full_name'},
    {'op':'NORMALIZE_CASE','column':'region'},
    {'op':'FILL_MISSING','column':'region','value':'Unknown'},
]:
    obs, rew, done, info = env.step(Action(**a))
    print(f"  {a['op']:22s} rew={rew.value:.3f} acc={obs.cell_accuracy:.3f}")
print(f"  FINAL={info.get('final_score', obs.cell_accuracy):.4f}")

# MEDIUM
print()
print('=== MEDIUM ===')
env.reset('medium')
for a in [
    {'op':'STRIP_WHITESPACE','column':'country'},
    {'op':'NORMALIZE_CASE','column':'country'},
    {'op':'PARSE_DATE','column':'order_date'},
    {'op':'DEDUP_ROWS','column':'order_id,line_item'},
]:
    obs, rew, done, info = env.step(Action(**a))
    print(f"  {a['op']:22s} rew={rew.value:.3f} acc={obs.cell_accuracy:.3f}")
print(f"  FINAL={info.get('final_score', obs.cell_accuracy):.4f}")

# HARD
print()
print('=== HARD ===')
env.reset('hard')
for a in [
    {'op':'EXTRACT_MONTH','column':'timestamp','value':'month'},
    {'op':'CONVERT_CURRENCY','column':'currency'},
    {'op':'GROUPBY_SUM','column':'month,country','value':'amount'},
    {'op':'RENAME_COLUMN','column':'amount','value':'total_revenue'},
]:
    obs, rew, done, info = env.step(Action(**a))
    print(f"  {a['op']:22s} rew={rew.value:.3f} acc={obs.cell_accuracy:.3f} schema={obs.schema_score:.3f}")
print(f"  FINAL={info.get('final_score', obs.cell_accuracy):.4f}")
print(f"  cols={obs.columns}")

# EXTREME
print()
print('=== EXTREME ===')
env.reset('extreme')
for a in [
    {'op':'EXTRACT_JSON','column':'metadata_json','value':'source','pattern':'lead_source'},
    {'op':'CAST_NUMERIC','column':'revenue_usd'},
    {'op':'PII_REDACT','column':'contact_email'},
    {'op':'DROP_OUTLIERS','column':'revenue_usd'},
    {'op':'NORMALIZE_CASE','column':'region'},
    {'op':'GROUPBY_SUM','column':'lead_source,region','value':'revenue_usd'},
    {'op':'RENAME_COLUMN','column':'revenue_usd','value':'total_revenue'},
]:
    obs, rew, done, info = env.step(Action(**a))
    print(f"  {a['op']:22s} rew={rew.value:.3f} acc={obs.cell_accuracy:.3f} rows={obs.row_count}")
print(f"  FINAL={info.get('final_score', obs.cell_accuracy):.4f}")
print(f"  cols={obs.columns}")

print()
print('=== ALL DONE ===')
