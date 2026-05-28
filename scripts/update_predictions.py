"""Re-run XGBoost predictions on all parquets and save in-place."""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.driver_soul import add_predictions_to_race

parquets = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "cache", "*.parquet")))
print(f"Updating {len(parquets)} parquets...")
for path in parquets:
    slug = os.path.basename(path).replace(".parquet", "")
    df = pd.read_parquet(path)
    df = add_predictions_to_race(df)
    df.to_parquet(path, index=False)
    n_pit = df["pit_prob"].nunique()
    n_gain = df["position_gain_prob"].nunique()
    print(f"  {slug}: pit_prob unique={n_pit}  gain_prob unique={n_gain}  rows={len(df)}")

print("Done.")
