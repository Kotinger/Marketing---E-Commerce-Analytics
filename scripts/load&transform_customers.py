# load&transform_customers.py
# Что делает: грузит customers.csv, проверяет качество, делает age/tenure фичи, сохраняет customers_snap.csv.

from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data") / "customers.csv"
OUT_DIR = Path("data/snapshot")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_nrows(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


if not SRC.exists():
    print(f"Файл {SRC} не найден.")
    raise SystemExit(1)

df = load_csv_nrows(SRC)
"""Смотрю: инфу типы данных, кол-во строк/столбцов, пропуски.
Все понятно, пропусков нет."""
#df.info()
#print(df.isna().sum())
"""Проверяю дубликаты и уникальность customer_id.
Дублей нет, 1 строка = 1 клиент."""
#print(df.duplicated().sum())
#print(df["customer_id"].nunique(), len(df))

df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
#print(df["signup_date"].isna().sum())

"""Возраст от 18 до 70. Делаю возрастные группы — так удобнее смотреть когорты."""
#print(df["age"].describe())
age_bins = [17, 24, 34, 44, 54, 70]
age_labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

"""Считаю tenure от signup до max даты в данных (как снимок на конец периода).
Потом режу на корзины <6m / 6-12m / 1-2y / 2y+."""
ref_date = df["signup_date"].max()
df["tenure_days"] = (ref_date - df["signup_date"]).dt.days
tenure_bins = [-1, 180, 365, 730, np.inf]
tenure_labels = ["<6m", "6-12m", "1-2y", "2y+"]
df["tenure_bucket"] = pd.cut(df["tenure_days"], bins=tenure_bins, labels=tenure_labels)

"""Год и месяц регистрации — пригодятся для когортного взгляда."""
df["signup_year"] = df["signup_date"].dt.year.astype("Int16")
df["signup_month"] = df["signup_date"].dt.to_period("M").astype(str)

"""Сохраняю промежуточный результат рядом с новыми колонками."""
to_insert = ["age_group", "tenure_days", "tenure_bucket", "signup_year", "signup_month"]
cols = [c for c in df.columns.tolist() if c not in to_insert]
pos = cols.index("age") + 1
new_cols = cols[:pos] + to_insert + cols[pos:]
df_reordered = df.loc[:, new_cols]

out_csv = OUT_DIR / "customers_snap.csv"
df_reordered.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df_reordered.head(5)[["customer_id", "signup_date", "age", "age_group", "tenure_bucket"]])
print(df_reordered.shape)
