# dtypes_campaigns.py
# Что делает: читает campaigns_snap.csv, выставляет даты/категории/типы, сохраняет parquet.

from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow 

SRC = Path("data/snapshot") / "campaigns_snap.csv"

def load_csv_nrows(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)

df= load_csv_nrows (SRC,   50)  

"""Даты"""
df = pd.read_csv("data/snapshot/campaigns_snap.csv", parse_dates=['start_date','end_date'], low_memory=False)
if not pd.api.types.is_datetime64_any_dtype(df['start_date']):
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce', infer_datetime_format=True)
if not pd.api.types.is_datetime64_any_dtype(df['end_date']):
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce', infer_datetime_format=True)
#print(df[['campaign_id','start_date','end_date']].head())
"""Количество дней"""
df['campaign_length_days'] = pd.to_numeric(df['campaign_length_days'], errors='coerce').astype('Int16')
print("Примеры:", df['campaign_length_days'].head().tolist())
"""Категории"""
cat_cols = ['channel', 'objective', 'target_segment', 'length_cat_fixed', 'season_majority']
for c in cat_cols:
    if c in df.columns and not pd.api.types.is_categorical_dtype(df[c]):
        df[c] = df[c].astype('category')
#print(df[ [c for c in cat_cols if c in df.columns] ].dtypes)
#for c in cat_cols:
#    if c in df.columns:
#        print(f"\nValue counts для {c}:")
#        print(df[c].value_counts(dropna=False))

out_pq = Path("data/snapshot") / "campaigns_snap.parquet"
df.to_parquet(out_pq, index=False)
print("Saved parquet:", out_pq)

df_check = pd.read_parquet(out_pq)
print("DTypes after reload:")
print(df_check.dtypes)
