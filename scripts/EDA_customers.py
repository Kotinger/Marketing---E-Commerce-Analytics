#!/usr/bin/env python3
# EDA_customers.py
# Что делает: Streamlit-дашборд по клиентам (когорты signup, age, loyalty, acquisition). 

from pathlib import Path
import datetime
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning, message="is_categorical_dtype")

# Страница — широкая
st.set_page_config(page_title="EDA — customers", layout="wide")

# Путь к данным
SRC = Path("data") / "snapshot" / "customers_snap.parquet"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    if "signup_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["signup_date"]):
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    return df


def apply_nice_style():
    if SEABORN_AVAILABLE:
        try:
            sns.set_theme(style="darkgrid")
            return
        except Exception:
            pass
    for name in ("ggplot", "bmh", "classic"):
        if name in plt.style.available:
            plt.style.use(name)
            return


def cats(col):
    if col in df.columns:
        try:
            return list(df[col].cat.categories)
        except Exception:
            return sorted(df[col].dropna().unique().tolist())
    return []


def make_arrow_safe(df_in: pd.DataFrame) -> pd.DataFrame:
    df2 = df_in.copy()
    for col in df2.columns:
        s = df2[col]
        try:
            if pd.api.types.is_extension_array_dtype(s.dtype):
                if pd.api.types.is_integer_dtype(s.dtype):
                    df2[col] = s.astype("float")
                    continue
                df2[col] = s.astype(object)
                continue
        except Exception:
            pass
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_datetime64_any_dtype(s.dtype):
            continue
        if pd.api.types.is_object_dtype(s.dtype):
            df2[col] = s.where(s.notnull(), None).astype(object)
    return df2


if not SRC.exists():
    st.error(f"Файл не найден: {SRC}")
    st.stop()

df = load_data(SRC)
apply_nice_style()

st.sidebar.header("Фильтры")
min_date = df["signup_date"].min().date()
max_date = df["signup_date"].max().date()
date_range = st.sidebar.date_input("Диапазон signup_date", value=(min_date, max_date))
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

sel_country = st.sidebar.multiselect("Country", options=cats("country"), default=cats("country"))
sel_gender = st.sidebar.multiselect("Gender", options=cats("gender"), default=cats("gender"))
sel_tier = st.sidebar.multiselect("Loyalty tier", options=cats("loyalty_tier"), default=cats("loyalty_tier"))
sel_acq = st.sidebar.multiselect("Acquisition", options=cats("acquisition_channel"), default=cats("acquisition_channel"))
sel_age = st.sidebar.multiselect("Age group", options=cats("age_group"), default=cats("age_group"))
age_lo, age_hi = st.sidebar.slider("Age", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
st.sidebar.caption("Чисто визуальный дашборд — ничего не сохраняет")

mask = (
    (df["signup_date"].dt.date >= start_date)
    & (df["signup_date"].dt.date <= end_date)
    & df["country"].isin(sel_country)
    & df["gender"].isin(sel_gender)
    & df["loyalty_tier"].isin(sel_tier)
    & df["acquisition_channel"].isin(sel_acq)
    & df["age_group"].isin(sel_age)
    & (df["age"] >= age_lo)
    & (df["age"] <= age_hi)
)
df_f = df.loc[mask].copy()

st.title("Customers EDA")
st.markdown("Лаконичный дашборд — фильтруй слева, смотри графики справа")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df_f):,}")
k2.metric("Avg age", f"{df_f['age'].mean():.1f}" if len(df_f) else "n/a")
k3.metric("Median tenure (days)", f"{df_f['tenure_days'].median():.0f}" if len(df_f) else "n/a")
top_tier = df_f["loyalty_tier"].value_counts().index[0] if len(df_f) else "n/a"
k4.metric("Top loyalty tier", top_tier)

st.markdown("---")
left, right = st.columns((2, 1))

with left:
    st.subheader("Signups by month")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(10, 3))
        ts = df_f.groupby(df_f["signup_date"].dt.to_period("M"))["customer_id"].count()
        ts.index = ts.index.to_timestamp()
        ax.plot(ts.index, ts.values, marker="o", color="#1f77b4")
        ax.set_ylabel("signups")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Age by loyalty tier")
    if len(df_f) and SEABORN_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 3))
        order = list(df_f["loyalty_tier"].cat.categories) if isinstance(df_f["loyalty_tier"].dtype, pd.CategoricalDtype) else None
        sns.boxplot(x="loyalty_tier", y="age", data=df_f, order=order, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with right:
    st.subheader("By country")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(6, 3))
        df_f["country"].value_counts().plot(kind="barh", ax=ax, color="#636EFA")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Acquisition channel")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(6, 3))
        df_f["acquisition_channel"].value_counts().plot(kind="barh", ax=ax, color="#2ca02c")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.subheader("Loyalty × Age group (counts)")
if len(df_f):
    ct = pd.crosstab(df_f["loyalty_tier"], df_f["age_group"])
    if PLOTLY_AVAILABLE:
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")
    else:
        st.dataframe(ct)

st.subheader("Filtered rows (preview)")
st.dataframe(make_arrow_safe(df_f.head(500)).reset_index(drop=True))
st.caption(f"Data preview from {SRC} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
