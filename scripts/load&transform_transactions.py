# load&transform_transactions.py
# Что делает: грузит transactions.csv, чистит/добавляет флаги и временные фичи, сохраняет transactions_snap.csv.

from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data") / "transactions.csv"
OUT_DIR = Path("data/snapshot")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_nrows(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


if not SRC.exists():
    print(f"Файл {SRC} не найден.")
    raise SystemExit(1)

df = load_csv_nrows(SRC)
"""Смотрю: инфо, пропуски, дубликаты.
Дублей нет. ~10% NA в product_id и gross_revenue — похоже одни и те же строки.
refund_flag=1 даёт отрицательный gross_revenue. campaign_id=0 = без кампании."""
#df.info()
#print(df.isna().sum())
#print(df.duplicated().sum())
#print((df["product_id"].isna() & df["gross_revenue"].isna()).sum())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#print(df["timestamp"].isna().sum())

"""Делаю флаги — так проще фильтровать на дашборде и в join."""
df["has_product"] = df["product_id"].notna().astype(int)
df["is_refund"] = (df["refund_flag"] == 1).astype(int)
df["has_campaign"] = (df["campaign_id"] > 0).astype(int)
df["has_discount"] = (df["discount_applied"] > 0).astype(int)

"""net_revenue оставляю = gross_revenue (у refund уже минус).
missing_revenue помечаю отдельно, чтоб не терять информацию."""
df["net_revenue"] = df["gross_revenue"]
df["missing_revenue"] = df["gross_revenue"].isna().astype(int)

"""Временные фичи: день/месяц/день недели/час."""
df["tx_date"] = df["timestamp"].dt.date.astype(str)
df["tx_month"] = df["timestamp"].dt.to_period("M").astype(str)
df["tx_dow"] = df["timestamp"].dt.day_name()
df["tx_hour"] = df["timestamp"].dt.hour.astype("Int8")

"""Скидки уже дискретные 0 / 0.05 / 0.1 / 0.15 / 0.2 — перевожу в читаемые корзины."""
df["discount_bin"] = pd.Categorical(
    df["discount_applied"].map({0.0: "0%", 0.05: "5%", 0.1: "10%", 0.15: "15%", 0.2: "20%"})
)

"""Сохраняю промежуточный результат."""
to_insert = [
    "has_product", "is_refund", "has_campaign", "has_discount",
    "net_revenue", "missing_revenue",
    "tx_date", "tx_month", "tx_dow", "tx_hour", "discount_bin",
]
cols = [c for c in df.columns.tolist() if c not in to_insert]
pos = cols.index("gross_revenue") + 1
new_cols = cols[:pos] + to_insert + cols[pos:]
df_reordered = df.loc[:, new_cols]

out_csv = OUT_DIR / "transactions_snap.csv"
df_reordered.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df_reordered.head(5)[["transaction_id", "gross_revenue", "is_refund", "has_campaign", "discount_bin"]])
print(df_reordered.shape)
print("missing product/revenue %:", round(df_reordered["missing_revenue"].mean() * 100, 2))
print("refund rate %:", round(df_reordered["is_refund"].mean() * 100, 2))
