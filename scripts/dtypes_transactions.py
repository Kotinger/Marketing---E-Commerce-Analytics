# dtypes_transactions.py
# Что делает: читает transactions_snap.csv, ставит nullable типы (product/revenue NA), категории, сохраняет parquet.

from pathlib import Path
import pandas as pd

SRC = Path("data/snapshot") / "transactions_snap.csv"


def load_csv_nrows(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


df = load_csv_nrows(SRC, 50)

"""Даты"""
df = pd.read_csv(SRC, parse_dates=["timestamp"], low_memory=False)
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

"""Числа — product_id / gross_revenue nullable (~10% NA), поэтому Int/float с NA."""
df["transaction_id"] = pd.to_numeric(df["transaction_id"], errors="coerce").astype("Int32")
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int32")
df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int16")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int8")
df["discount_applied"] = pd.to_numeric(df["discount_applied"], errors="coerce").astype("float32")
df["gross_revenue"] = pd.to_numeric(df["gross_revenue"], errors="coerce").astype("float32")
df["net_revenue"] = pd.to_numeric(df["net_revenue"], errors="coerce").astype("float32")
df["campaign_id"] = pd.to_numeric(df["campaign_id"], errors="coerce").astype("Int8")
df["refund_flag"] = pd.to_numeric(df["refund_flag"], errors="coerce").astype("Int8")
df["tx_hour"] = pd.to_numeric(df["tx_hour"], errors="coerce").astype("Int8")

for c in ["has_product", "is_refund", "has_campaign", "has_discount", "missing_revenue"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")

"""Категории"""
cat_cols = ["tx_month", "tx_dow", "discount_bin", "tx_date"]
for c in cat_cols:
    if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype("category")

out_pq = Path("data/snapshot") / "transactions_snap.parquet"
df.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

df_check = pd.read_parquet(out_pq)
print("DTypes after reload:")
print(df_check.dtypes)
print("shape:", df_check.shape)
print("NA product_id:", int(df_check["product_id"].isna().sum()))
