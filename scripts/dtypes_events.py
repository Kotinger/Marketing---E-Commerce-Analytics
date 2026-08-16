# dtypes_events.py
# Что делает: дожимает типы в events_snap.parquet (даты/nullable id/категории) и перезаписывает parquet.

from pathlib import Path
import pandas as pd

"""Parquet уже пишет load&transform_events.py.
Тут только типы — как в остальных dtypes_*."""
SRC = Path("data/snapshot") / "events_snap.parquet"

if not SRC.exists():
    SRC = Path("data/snapshot") / "events_snap_sample.csv"

df = pd.read_parquet(SRC) if SRC.suffix == ".parquet" else pd.read_csv(SRC, low_memory=False)

"""Даты"""
if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

"""Числа — product_id nullable"""
df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int32")
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int32")
df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce").astype("Int32")
df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int16")
df["campaign_id"] = pd.to_numeric(df["campaign_id"], errors="coerce").astype("Int8")
df["session_duration_sec"] = pd.to_numeric(df["session_duration_sec"], errors="coerce").astype("float32")
df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").astype("Int8")

for c in ["has_product", "has_campaign", "has_device", "is_purchase", "is_bounce"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")

"""Категории"""
cat_cols = [
    "event_type", "device_type", "traffic_source", "page_category",
    "experiment_group", "duration_bucket", "event_month", "event_date",
]
for c in cat_cols:
    if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype("category")

out_pq = Path("data/snapshot") / "events_snap.parquet"
df.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

df_check = pd.read_parquet(out_pq)
print("DTypes after reload:")
print(df_check.dtypes)
print("shape:", df_check.shape)
print("NA product_id:", int(df_check["product_id"].isna().sum()))
print("NA device_type:", int(df_check["device_type"].isna().sum()))
