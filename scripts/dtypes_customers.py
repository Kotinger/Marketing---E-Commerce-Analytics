# dtypes_customers.py
# Что делает: читает customers_snap.csv, выставляет даты/числа/категории, сохраняет customers_snap.parquet.

from pathlib import Path
import pandas as pd

SRC = Path("data/snapshot") / "customers_snap.csv"


def load_csv_nrows(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


df = load_csv_nrows(SRC, 50)

"""Даты"""
df = pd.read_csv(SRC, parse_dates=["signup_date"], low_memory=False)
if not pd.api.types.is_datetime64_any_dtype(df["signup_date"]):
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

"""Числа"""
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int32")
df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int8")
df["tenure_days"] = pd.to_numeric(df["tenure_days"], errors="coerce").astype("Int16")
df["signup_year"] = pd.to_numeric(df["signup_year"], errors="coerce").astype("Int16")

"""Категории"""
cat_cols = [
    "country", "gender", "loyalty_tier", "acquisition_channel",
    "age_group", "tenure_bucket", "signup_month",
]
for c in cat_cols:
    if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype("category")

out_pq = Path("data/snapshot") / "customers_snap.parquet"
df.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

df_check = pd.read_parquet(out_pq)
print("DTypes after reload:")
print(df_check.dtypes)
print("shape:", df_check.shape)
