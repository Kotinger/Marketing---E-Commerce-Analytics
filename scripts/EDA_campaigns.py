# EDA_campaigns.py
# Что делает: Streamlit-дашборд по campaigns (фильтры, KPI, графики). 


from pathlib import Path
import datetime
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Опционально plotly/seaborn
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns  # type: ignore
    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False

# Подавляем только конкретное предупреждение pandas4 (исправлено в коде, но на всякий случай)
warnings.filterwarnings("ignore", category=FutureWarning, message="is_categorical_dtype")

# Страница — широкая
st.set_page_config(page_title="EDA — campaigns", layout="wide")

# Путь к данным
SRC = Path("data") / "snapshot" / "campaigns_snap.parquet"

# ---------------- загрузка ----------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=['start_date', 'end_date'], low_memory=False)
    if 'start_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['start_date']):
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    if 'end_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['end_date']):
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    return df

if not SRC.exists():
    st.error(f"Файл не найден: {SRC}")
    st.stop()

df = load_data(SRC)

# ---------------- стили графиков ----------------
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
    plt.style.use("default")

apply_nice_style()

# -------------- утилиты --------------
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
                if pd.api.types.is_bool_dtype(s.dtype) or (hasattr(s.dtype, "name") and s.dtype.name == "boolean"):
                    df2[col] = s.where(s.notnull(), None).astype(object)
                    continue
                df2[col] = s.astype(object)
                continue
        except Exception:
            pass

        try:
            is_cat = isinstance(s.dtype, pd.CategoricalDtype)
        except Exception:
            is_cat = False

        if pd.api.types.is_numeric_dtype(s.dtype) or is_cat or pd.api.types.is_datetime64_any_dtype(s.dtype):
            continue

        if pd.api.types.is_object_dtype(s.dtype):
            def _to_native(v):
                if pd.isna(v):
                    return None
                if isinstance(v, np.generic):
                    try:
                        return v.item()
                    except Exception:
                        return v
                try:
                    import pandas as _pd
                    if isinstance(v, _pd.Timestamp):
                        return v.to_pydatetime()
                    if isinstance(v, _pd.Period):
                        return str(v)
                except Exception:
                    pass
                if isinstance(v, (list, dict, tuple, set)):
                    try:
                        return str(v)
                    except Exception:
                        return v
                return v
            try:
                df2[col] = s.map(_to_native).astype(object)
            except Exception:
                df2[col] = s.where(s.notnull(), None).astype(object)
    return df2

# ----------------- Sidebar filters -----------------
st.sidebar.header("Фильтры")

# date range
if 'start_date' in df.columns:
    min_date = df['start_date'].min().date()
    max_date = df['start_date'].max().date()
    date_range = st.sidebar.date_input("Диапазон по start_date", value=(min_date, max_date))
    if isinstance(date_range, (list, tuple)):
        start_date, end_date = date_range[0], date_range[1]
    else:
        start_date = end_date = date_range
else:
    start_date = end_date = None

# categories
sel_objective = st.sidebar.multiselect("Objective", options=cats('objective'), default=cats('objective'))
sel_channel = st.sidebar.multiselect("Channel", options=cats('channel'), default=cats('channel'))
sel_length = st.sidebar.multiselect("Length category", options=cats('length_cat_fixed'), default=cats('length_cat_fixed'))
sel_season = st.sidebar.multiselect("Season", options=cats('season_majority'), default=cats('season_majority'))

# expected_uplift
if 'expected_uplift' in df.columns:
    lo = float(df['expected_uplift'].min(skipna=True))
    hi = float(df['expected_uplift'].max(skipna=True))
    u_lo, u_hi = st.sidebar.slider("expected_uplift", min_value=lo, max_value=hi, value=(lo, hi))
else:
    u_lo = u_hi = None

st.sidebar.markdown("---")
st.sidebar.caption("профиль кампаний")

# ----------------- apply filters -----------------
mask = pd.Series(True, index=df.index)
if start_date is not None and end_date is not None and 'start_date' in df.columns:
    mask &= (df['start_date'].dt.date >= start_date) & (df['start_date'].dt.date <= end_date)
if sel_objective:
    mask &= df['objective'].isin(sel_objective)
if sel_channel:
    mask &= df['channel'].isin(sel_channel)
if sel_length:
    mask &= df['length_cat_fixed'].isin(sel_length)
if sel_season:
    mask &= df['season_majority'].isin(sel_season)
if u_lo is not None and u_hi is not None and 'expected_uplift' in df.columns:
    mask &= (df['expected_uplift'] >= u_lo) & (df['expected_uplift'] <= u_hi)

df_f = df.loc[mask].copy()

# ----------------- Header / KPIs -----------------
st.title("Campaigns EDA")
st.markdown("Профиль кампаний")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (filtered)", f"{len(df_f):,}")
if 'expected_uplift' in df_f.columns and df_f['expected_uplift'].dropna().size > 0:
    s = df_f['expected_uplift'].dropna().astype(float)
    k2.metric("Uplift mean", f"{s.mean():.4f}")
    k3.metric("Uplift median", f"{s.median():.4f}")
else:
    k2.metric("Uplift mean", "n/a"); k3.metric("Uplift median", "n/a")

if 'channel' in df_f.columns and 'expected_uplift' in df_f.columns and len(df_f)>0:
    try:
        best_chan = df_f.groupby('channel')['expected_uplift'].median().sort_values(ascending=False).index[0]
        k4.metric("Best channel", best_chan)
    except Exception:
        k4.metric("Best channel", "n/a")
else:
    k4.metric("Best channel", "n/a")

st.markdown("---")

# ----------------- Charts -----------------
left, right = st.columns((2,1))

with left:
    st.subheader("Campaigns by month")
    if 'start_date' in df_f.columns and len(df_f) > 0:
        fig, ax = plt.subplots(figsize=(10,3))
        ts = df_f.groupby(df_f['start_date'].dt.to_period("M"))['campaign_id'].count()
        ts.index = ts.index.to_timestamp()
        ax.plot(ts.index, ts.values, marker='o', color="#1f77b4")
        ax.set_ylabel("campaigns")
        ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Нет start_date / данных для временного ряда")

    st.subheader("Uplift by length category")
    if 'expected_uplift' in df_f.columns and 'length_cat_fixed' in df_f.columns:
        try:
            if SEABORN_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10,3))
                order = list(df_f['length_cat_fixed'].cat.categories) if isinstance(df_f['length_cat_fixed'].dtype, pd.CategoricalDtype) else None
                # убрал palette, чтобы не было FutureWarning
                sns.boxplot(x='length_cat_fixed', y='expected_uplift', data=df_f, order=order, ax=ax)
                ax.set_xlabel("")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(10,3))
                df_f.boxplot(column='expected_uplift', by='length_cat_fixed', ax=ax)
                ax.get_figure().suptitle("")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.write("Ошибка построения boxplot:", e)
    else:
        st.info("Требуются столбцы expected_uplift и length_cat_fixed")

with right:
    st.subheader("Uplift vs campaign length")
    if 'expected_uplift' in df_f.columns and 'campaign_length_days' in df_f.columns:
        if PLOTLY_AVAILABLE:
            try:
                fig = px.scatter(df_f, x='campaign_length_days', y='expected_uplift',
                                 color='length_cat_fixed' if 'length_cat_fixed' in df_f.columns else None,
                                 hover_data=['campaign_id'] if 'campaign_id' in df_f.columns else None,
                                 trendline="ols", height=420)
                # новое API: width='stretch' вместо use_container_width
                st.plotly_chart(fig, width='stretch')
            except Exception:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(df_f['campaign_length_days'].astype(float), df_f['expected_uplift'].astype(float), alpha=0.6)
                ax.set_xlabel("campaign_length_days"); ax.set_ylabel("expected_uplift")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(df_f['campaign_length_days'].astype(float), df_f['expected_uplift'].astype(float), alpha=0.6, color="#2ca02c")
            ax.set_xlabel("campaign_length_days"); ax.set_ylabel("expected_uplift")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Нужны expected_uplift и campaign_length_days")

    st.subheader("Counts by channel")
    if 'channel' in df_f.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        vc = df_f['channel'].value_counts().nlargest(15)
        vc.plot(kind='barh', ax=ax, color="#636EFA")
        ax.invert_yaxis()
        ax.set_xlabel("count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Нет данных по channel")

st.markdown("---")

# ----------------- Table -----------------
st.subheader("Filtered rows (preview)")
try:
    df_display = make_arrow_safe(df_f).reset_index(drop=True)
    # убрал use_container_width, чтобы не было предупреждений
    st.dataframe(df_display)
except Exception:
    st.error("Не удалось показать таблицу через st.dataframe")
    st.write(df_f.head(100))

st.caption(f"Data preview from {SRC} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")