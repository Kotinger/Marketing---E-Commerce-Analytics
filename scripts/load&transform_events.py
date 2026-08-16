# load&transform_events.py
# Что делает: грузит большой events.csv (~2M), нормализует traffic_source, добавляет флаги/корзины,
# сохраняет events_snap.parquet + лёгкий sample CSV (полный CSV слишком тяжёлый).

from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data") / "events.csv"
OUT_DIR = Path("data/snapshot")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_nrows(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


if not SRC.exists():
    print(f"Файл {SRC} не найден.")
    raise SystemExit(1)

"""Смотрю размер и пропуски.
~2M строк — pandas тянет. product_id ~10% NA (часто bounce), device_type ~2% NA.
traffic_source в разном регистре (Organic / ORGANIC) — надо нормализовать."""
df = load_csv_nrows(SRC)
#print(df.shape)
#print(df.isna().mean().round(3))
#print(df["traffic_source"].value_counts())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

"""Привожу traffic_source к одному виду через title()."""
df["traffic_source"] = (
    df["traffic_source"]
    .astype(str)
    .str.strip()
    .str.title()
)

"""Флаги для быстрых фильтров и дальнейшей агрегации в сессии."""
df["has_product"] = df["product_id"].notna().astype(int)
df["has_campaign"] = (df["campaign_id"] > 0).astype(int)
df["has_device"] = df["device_type"].notna().astype(int)
df["is_purchase"] = (df["event_type"] == "purchase").astype(int)
df["is_bounce"] = (df["event_type"] == "bounce").astype(int)

"""Длительность сессии режу на корзины — так проще смотреть распределения."""
#print(df["session_duration_sec"].describe())
dur_bins = [0, 30, 60, 120, 300, np.inf]
dur_labels = ["0-30s", "30-60s", "1-2m", "2-5m", "5m+"]
df["duration_bucket"] = pd.cut(
    df["session_duration_sec"], bins=dur_bins, labels=dur_labels, include_lowest=True
)

df["event_date"] = df["timestamp"].dt.date.astype(str)
df["event_month"] = df["timestamp"].dt.to_period("M").astype(str)
df["event_hour"] = df["timestamp"].dt.hour.astype("Int8")

to_insert = [
    "has_product", "has_campaign", "has_device",
    "is_purchase", "is_bounce", "duration_bucket",
    "event_date", "event_month", "event_hour",
]
cols = [c for c in df.columns.tolist() if c not in to_insert]
pos = cols.index("session_duration_sec") + 1
new_cols = cols[:pos] + to_insert + cols[pos:]
df_reordered = df.loc[:, new_cols]

"""Полный CSV тут не пишу — тяжело. Parquet основной + sample csv для быстрых проверок."""
out_pq = OUT_DIR / "events_snap.parquet"
df_reordered.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

sample = df_reordered.sample(n=min(50_000, len(df_reordered)), random_state=42)
out_csv = OUT_DIR / "events_snap_sample.csv"
sample.to_csv(out_csv, index=False)
print("Saved sample csv:", out_csv)

print(df_reordered.head(3)[["event_id", "event_type", "traffic_source", "duration_bucket", "has_campaign"]])
print(df_reordered.shape)
print("traffic_source after norm:", df_reordered["traffic_source"].value_counts().to_dict())
print("device NA %:", round(df_reordered["device_type"].isna().mean() * 100, 2))
