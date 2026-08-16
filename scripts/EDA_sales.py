# EDA_sales.py
# Что делает: Streamlit по продажам — transactions ⋈ customers ⋈ products ⋈ campaigns.
# Смотрю выручку, категории, каналы кампаний, loyalty. 

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

warnings.filterwarnings("ignore", category=FutureWarning)

# Страница — широкая
st.set_page_config(page_title="EDA — sales (joined)", layout="wide")

# Пути к snapshot
SNAP = Path("data") / "snapshot"


@st.cache_data
def load_sales() -> pd.DataFrame:
    """Джойню факты с измерениями. campaign_id=0 → No campaign."""
    tx = pd.read_parquet(SNAP / "transactions_snap.parquet")
    cust = pd.read_parquet(SNAP / "customers_snap.parquet")
    prod = pd.read_parquet(SNAP / "products_snap.parquet")
    camp = pd.read_parquet(SNAP / "campaigns_snap.parquet")

    if not pd.api.types.is_datetime64_any_dtype(tx["timestamp"]):
        tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")

    # campaign_id=0 → без кампании
    camp_cols = [
        "campaign_id", "channel", "objective", "target_segment",
        "expected_uplift", "length_cat_fixed", "season_majority",
    ]
    camp = camp[[c for c in camp_cols if c in camp.columns]].copy()
    camp = camp.rename(columns={
        "channel": "campaign_channel",
        "objective": "campaign_objective",
        "target_segment": "campaign_segment",
    })

    df = tx.merge(cust, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(
        prod[["product_id", "category", "brand", "base_price", "price_tier", "premium_flag"]],
        on="product_id",
        how="left",
    )
    df = df.merge(camp, on="campaign_id", how="left")

    df["campaign_channel"] = df["campaign_channel"].astype(object).fillna("No campaign")
    df["campaign_objective"] = df["campaign_objective"].astype(object).fillna("No campaign")
    return df


def apply_style():
    if SEABORN_AVAILABLE:
        try:
            sns.set_theme(style="darkgrid")
            return
        except Exception:
            pass
    if "ggplot" in plt.style.available:
        plt.style.use("ggplot")


def cats(series):
    try:
        return list(series.cat.categories)
    except Exception:
        return sorted([x for x in series.dropna().unique().tolist()])


def arrow_safe(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_extension_array_dtype(s.dtype) and pd.api.types.is_integer_dtype(s.dtype):
            out[col] = s.astype("float")
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(object)
    return out


needed = [
    "transactions_snap.parquet",
    "customers_snap.parquet",
    "products_snap.parquet",
    "campaigns_snap.parquet",
]
missing = [p for p in needed if not (SNAP / p).exists()]
if missing:
    st.error("Нет файлов: " + ", ".join(missing))
    st.stop()

df = load_sales()
apply_style()

st.sidebar.header("Фильтры")
min_d, max_d = df["timestamp"].min().date(), df["timestamp"].max().date()
dr = st.sidebar.date_input("Период", value=(min_d, max_d))
start_d, end_d = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dr, dr))

countries = cats(df["country"]) if "country" in df.columns else []
categories = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
channels = sorted(df["campaign_channel"].dropna().unique().tolist())
tiers = cats(df["loyalty_tier"]) if "loyalty_tier" in df.columns else []

sel_country = st.sidebar.multiselect("Country", countries, default=countries)
sel_cat = st.sidebar.multiselect("Product category", categories, default=categories)
sel_chan = st.sidebar.multiselect("Campaign channel", channels, default=channels)
sel_tier = st.sidebar.multiselect("Loyalty tier", tiers, default=tiers)
hide_missing = st.sidebar.checkbox("Скрыть missing product/revenue", value=True)
only_refund = st.sidebar.checkbox("Только refund", value=False)
st.sidebar.caption("Sales = transactions + dims")

mask = (df["timestamp"].dt.date >= start_d) & (df["timestamp"].dt.date <= end_d)
if sel_country:
    mask &= df["country"].isin(sel_country)
if sel_cat:
    if hide_missing:
        mask &= df["category"].isin(sel_cat)
    else:
        mask &= df["category"].isin(sel_cat) | df["category"].isna()
if sel_chan:
    mask &= df["campaign_channel"].isin(sel_chan)
if sel_tier:
    mask &= df["loyalty_tier"].isin(sel_tier)
if hide_missing:
    mask &= df["missing_revenue"] == 0
if only_refund:
    mask &= df["is_refund"] == 1

df_f = df.loc[mask].copy()
rev = df_f["net_revenue"].dropna()

st.title("Sales EDA (joined)")
st.markdown("Транзакции с клиентами, товарами и кампаниями")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", f"{len(df_f):,}")
k2.metric("Net revenue", f"{rev.sum():,.0f}" if len(rev) else "n/a")
k3.metric("AOV", f"{rev.mean():.1f}" if len(rev) else "n/a")
k4.metric("Unique customers", f"{df_f['customer_id'].nunique():,}" if len(df_f) else "n/a")

st.markdown("---")
left, right = st.columns((2, 1))

with left:
    st.subheader("Revenue by month")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(10, 3))
        ts = df_f.groupby(df_f["timestamp"].dt.to_period("M"))["net_revenue"].sum()
        ts.index = ts.index.to_timestamp()
        ax.plot(ts.index, ts.values, marker="o", color="#1f77b4")
        ax.set_ylabel("net_revenue")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Revenue by product category")
    if len(df_f) and df_f["category"].notna().any():
        cat_rev = df_f.groupby("category", dropna=True)["net_revenue"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(10, 3.2))
        cat_rev.plot(kind="barh", ax=ax, color="#636EFA")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with right:
    st.subheader("Revenue by campaign channel")
    if len(df_f):
        ch = df_f.groupby("campaign_channel")["net_revenue"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ch.plot(kind="barh", ax=ax, color="#2ca02c")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Loyalty mix (tx count)")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(6, 3))
        df_f["loyalty_tier"].value_counts().plot(kind="barh", ax=ax, color="#ff7f0e")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.subheader("AOV by country")
    if len(df_f):
        aov = df_f.groupby("country")["net_revenue"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        aov.plot(kind="bar", ax=ax, color="#1f77b4")
        ax.set_ylabel("avg net_revenue")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
with c2:
    st.subheader("Premium vs revenue")
    if len(df_f) and df_f["premium_flag"].notna().any():
        g = df_f.groupby("premium_flag", dropna=True).agg(
            tx=("transaction_id", "count"),
            revenue=("net_revenue", "sum"),
            aov=("net_revenue", "mean"),
        )
        st.dataframe(g.round(2))

st.subheader("Campaign objective × product category (revenue)")
if len(df_f):
    pivot = pd.pivot_table(
        df_f,
        values="net_revenue",
        index="campaign_objective",
        columns="category",
        aggfunc="sum",
        fill_value=0,
    )
    if PLOTLY_AVAILABLE and pivot.size:
        fig = px.imshow(pivot, text_auto=".0f", aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")
    else:
        st.dataframe(pivot.round(0))

st.subheader("Top brands by revenue")
if len(df_f) and df_f["brand"].notna().any():
    top = df_f.groupby("brand")["net_revenue"].sum().nlargest(12)
    fig, ax = plt.subplots(figsize=(10, 3))
    top.plot(kind="bar", ax=ax, color="#9467bd")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Preview")
cols_show = [
    c for c in [
        "transaction_id", "timestamp", "customer_id", "country", "loyalty_tier",
        "product_id", "category", "brand", "premium_flag",
        "net_revenue", "discount_bin", "is_refund",
        "campaign_id", "campaign_channel", "campaign_objective",
    ] if c in df_f.columns
]
st.dataframe(arrow_safe(df_f[cols_show].head(400)).reset_index(drop=True))
st.caption(f"Joined sales view — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
