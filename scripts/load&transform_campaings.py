# load&transform_campaings.py
# Что делает: грузит campaigns.csv, проверяет качество, добавляет длительность/сезон/категорию длины, сохраняет snapshot CSV.

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


SRC = Path("data") / "campaigns.csv"

def load_csv_nrows(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)
if not SRC.exists():
    print(f"Файл {SRC} не найден.")
df = load_csv_nrows(SRC, nrows=50)
"""Смотрю: инфу типы данных, количество строк и столбцов, количество пропусков в каждом столбце и есть ли пропуски. 
Все понятно, нет пропусков."""
#df.info()
#print(df)    
#print(df.isna().sum()) 
"""Проверяю есть ли дубликаты в данных. 
Дублий нет"""
#dup = df.duplicated().sum()
#print(f"Количество дубликатов: {dup}")
"""Проверяю количество уникальных значений в столбце campaign_id. 
1 строка = 1 компания"""
#if 'campaign_id' in df.columns:
#    print( df['campaign_id'].nunique())
"""Проверяю парсятся ли даты в столбцах start_date и end_date и есть ли отрицательные длитеольности. 
Все норм, отрицательной длительности тож нет"""
df ['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df ['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
#print(df['start_date'].isna().sum(), df['end_date'].isna().sum())
df['campaign_length_days'] = (df['end_date'] - df['start_date']).dt.days
#print("start> end:", (df['campaign_length_days'] < 0).sum())
"""Посмотрел результат count/mean/std/min/percentiles/max. 
Прихожу к выводу, что создам доп колонки : с длительностью кампании в днях,   категории по квартелям, сезоны"""
#print (df['campaign_length_days'].describe())
bins = [ 0, 37, 68, np.inf ] 
labels = ['small','medium','large']
df['length_cat_fixed'] = pd.cut(df['campaign_length_days'], bins=bins, labels=labels)
month_to_season = {
    12: 'winter', 1: 'winter', 2: 'winter',
     3: 'spring', 4: 'spring', 5: 'spring',
     6: 'summer', 7: 'summer', 8: 'summer',
     9: 'autumn',10: 'autumn',11: 'autumn'
}
def season_by_majority(row, month_to_season):
    s, e = row['start_date'], row['end_date']
    if pd.isna(s) or pd.isna(e):
        return np.nan
    days = pd.date_range(s, e)
    seasons = [month_to_season[m] for m in days.month]
    return pd.Series(seasons).value_counts().idxmax()
df['season_majority'] = df.apply(lambda r: season_by_majority(r, month_to_season), axis=1)
#print(df[['campaign_length_days','length_cat_fixed', 'season_majority']])
"""Сохраняю промежуточный результат"""
to_insert= ['campaign_length_days','length_cat_fixed', 'season_majority']
missing = [c for c in to_insert if c not in df.columns]
cols=[c for c in df.columns.tolist() if c not in to_insert]
pos= cols.index('end_date')+1
new_cols = cols[:pos] + to_insert + cols[pos:]
df_reordered = df.loc[:, new_cols]
out_csv = Path("data/snapshot") / "campaigns_snap.csv"
df_reordered.to_csv(out_csv, index=False)
print(df.info(),df_reordered.head(5)[ ['end_date'] + to_insert ])