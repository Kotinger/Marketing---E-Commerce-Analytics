# load&transform_products.py
# Что делает: лёгкий dim-prep products.csv → price_tier/season/premium_flag → products_snap.csv.
# Отдельный EDA по товарам не делаю: интерес появляется в EDA_sales через join.

from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data") / "products.csv"
OUT_DIR = Path("data/snapshot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    print(f"Файл {SRC} не найден.")
    raise SystemExit(1)

df = pd.read_csv(SRC, low_memory=False)
"""Смотрю: инфо, пропуски, дубликаты.
Пропусков нет, дублей нет, 1 строка = 1 product_id.
Категорий 6, брендов 100, is_premium ровно 50/50."""
#df.info()
#print(df.isna().sum())
#print(df.duplicated().sum())
#print(df["category"].value_counts())

df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
#print(df["launch_date"].isna().sum())

"""Цена сильно разъезжается. Режу base_price по квартилям → price_tier."""
#print(df["base_price"].describe())
q1, q2, q3 = df["base_price"].quantile([0.25, 0.5, 0.75]).tolist()
df["price_tier"] = pd.cut(
    df["base_price"],
    bins=[0, q1, q2, q3, np.inf],
    labels=["budget", "mid", "upper", "premium_price"],
    include_lowest=True,
)

"""is_premium уже 0/1 — делаю читаемый флаг + год/сезон запуска."""
df["premium_flag"] = np.where(df["is_premium"] == 1, "premium", "standard")
df["launch_year"] = df["launch_date"].dt.year.astype("Int16")

month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
df["launch_season"] = df["launch_date"].dt.month.map(month_to_season)

"""Сохраняю dim-snapshot. Дальше товар смотрю только в sales-join."""
out_csv = OUT_DIR / "products_snap.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv, df.shape)
print(df.head(5)[["product_id", "category", "base_price", "price_tier", "premium_flag"]])
