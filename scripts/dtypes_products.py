# dtypes_products.py
# Что делает: читает products_snap.csv, выставляет даты/числа/категории, сохраняет products_snap.parquet.

from pathlib import Path
import pandas as pd

SRC = Path("data/snapshot") / "products_snap.csv"


def load_csv_nrows(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


df = load_csv_nrows(SRC, 50)

"""Даты"""
df = pd.read_csv(SRC, parse_dates=["launch_date"], low_memory=False)
if not pd.api.types.is_datetime64_any_dtype(df["launch_date"]):
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")

"""Числа / флаги"""
df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int16")
df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce").astype("float32")
df["is_premium"] = pd.to_numeric(df["is_premium"], errors="coerce").astype("Int8")
df["launch_year"] = pd.to_numeric(df["launch_year"], errors="coerce").astype("Int16")

"""Категории"""
cat_cols = ["category", "brand", "price_tier", "premium_flag", "launch_season"]
for c in cat_cols:
    if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype("category")

out_pq = Path("data/snapshot") / "products_snap.parquet"
df.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

df_check = pd.read_parquet(out_pq)
print("DTypes after reload:")
print(df_check.dtypes)
print("shape:", df_check.shape)
