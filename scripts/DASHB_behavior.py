# EDA_behavior.py
# Что делает: Streamlit по поведению — session funnel / traffic / device / experiment + A/B статтест.
# Читает sessions_snap.parquet (не сырые 2M events). 

from pathlib import Path
import datetime
import sys
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# функции статтеста из соседнего скрипта
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_test_sessions import run_ab_analysis  # noqa: E402

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
st.set_page_config(page_title="EDA — behavior (sessions)", layout="wide")

# Пути к данным
SRC = Path("data") / "snapshot" / "sessions_snap.parquet"
CAMP = Path("data") / "snapshot" / "campaigns_snap.parquet"


@st.cache_data
def load_sessions() -> pd.DataFrame:
    s = pd.read_parquet(SRC)
    if not pd.api.types.is_datetime64_any_dtype(s["session_start"]):
        s["session_start"] = pd.to_datetime(s["session_start"], errors="coerce")
    if CAMP.exists():
        camp = pd.read_parquet(CAMP)[["campaign_id", "channel", "objective"]].rename(
            columns={"channel": "campaign_channel", "objective": "campaign_objective"}
        )
        s = s.merge(camp, on="campaign_id", how="left")
        s["campaign_channel"] = s["campaign_channel"].astype(object).fillna("No campaign")
    else:
        s["campaign_channel"] = np.where(s["campaign_id"] > 0, "Campaign", "No campaign")
    return s


def apply_style():
    if SEABORN_AVAILABLE:
        try:
            sns.set_theme(style="darkgrid")
            return
        except Exception:
            pass
    if "ggplot" in plt.style.available:
        plt.style.use("ggplot")


def cats(col, df):
    if col not in df.columns:
        return []
    try:
        return list(df[col].cat.categories)
    except Exception:
        return sorted([x for x in df[col].dropna().unique().tolist()])


def funnel_counts(df_in: pd.DataFrame) -> pd.DataFrame:
    """Считаю сколько сессий дошли хотя бы до шага view→click→cart→purchase."""
    steps = [
        ("view", "has_view"),
        ("click", "has_click"),
        ("add_to_cart", "has_add_to_cart"),
        ("purchase", "has_purchase"),
    ]
    rows = []
    for name, col in steps:
        n = int(df_in[col].sum()) if col in df_in.columns else 0
        rows.append({"step": name, "sessions": n})
    out = pd.DataFrame(rows)
    if len(out) and out.loc[0, "sessions"]:
        out["rate_from_view"] = out["sessions"] / out.loc[0, "sessions"]
        out["step_conv"] = out["sessions"] / out["sessions"].shift(1)
        out.loc[0, "step_conv"] = 1.0
    return out


def arrow_safe(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_extension_array_dtype(s.dtype) and pd.api.types.is_integer_dtype(s.dtype):
            out[col] = s.astype("float")
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(object)
    return out


if not SRC.exists():
    st.error(f"Нет {SRC}. Запусти scripts/build_sessions.py")
    st.stop()

df = load_sessions()
apply_style()

st.sidebar.header("Фильтры")
min_d, max_d = df["session_start"].min().date(), df["session_start"].max().date()
dr = st.sidebar.date_input("Период", value=(min_d, max_d))
start_d, end_d = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dr, dr))

sel_device = st.sidebar.multiselect("Device", cats("device_type", df), default=cats("device_type", df))
sel_src = st.sidebar.multiselect("Traffic", cats("traffic_source", df), default=cats("traffic_source", df))
sel_exp = st.sidebar.multiselect("Experiment", cats("experiment_group", df), default=cats("experiment_group", df))
only_camp = st.sidebar.checkbox("Только с campaign", value=False)
st.sidebar.caption("Behavior = session aggregates from events")

mask = (df["session_start"].dt.date >= start_d) & (df["session_start"].dt.date <= end_d)
if sel_device:
    mask &= df["device_type"].isin(sel_device)
if sel_src:
    mask &= df["traffic_source"].isin(sel_src)
if sel_exp:
    mask &= df["experiment_group"].isin(sel_exp)
if only_camp:
    mask &= df["has_campaign"] == 1

df_f = df.loc[mask].copy()
funnel = funnel_counts(df_f)

st.title("Behavior EDA (sessions)")
st.markdown("Воронка, трафик, device и A/B — на уровне сессий")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sessions", f"{len(df_f):,}")
k2.metric("CR (purchase)", f"{(df_f['has_purchase'].mean()*100):.2f}%" if len(df_f) else "n/a")
k3.metric("Bounce sessions", f"{(df_f['has_bounce'].mean()*100):.2f}%" if len(df_f) else "n/a")
k4.metric("Median duration (s)", f"{df_f['duration_sec'].median():.0f}" if len(df_f) else "n/a")

st.markdown("---")
left, right = st.columns((2, 1))

with left:
    st.subheader("Funnel (sessions reaching step)")
    if len(funnel):
        if PLOTLY_AVAILABLE:
            fig = px.funnel(funnel, x="sessions", y="step", height=360)
            st.plotly_chart(fig, width="stretch")
        else:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.barh(funnel["step"][::-1], funnel["sessions"][::-1], color="#636EFA")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.dataframe(funnel.round(4))

    st.subheader("Sessions by month")
    if len(df_f):
        fig, ax = plt.subplots(figsize=(10, 3))
        ts = df_f.groupby(df_f["session_start"].dt.to_period("M"))["session_id"].count()
        ts.index = ts.index.to_timestamp()
        ax.plot(ts.index, ts.values, marker="o", color="#1f77b4")
        ax.set_ylabel("sessions")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with right:
    st.subheader("CR by traffic source")
    if len(df_f):
        g = df_f.groupby("traffic_source", observed=True).agg(
            sessions=("session_id", "count"),
            cr=("has_purchase", "mean"),
            bounce=("has_bounce", "mean"),
        ).sort_values("cr", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        g["cr"].plot(kind="barh", ax=ax, color="#2ca02c")
        ax.invert_yaxis()
        ax.set_xlabel("purchase rate")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("CR by device")
    if len(df_f):
        g = df_f.groupby("device_type", observed=True)["has_purchase"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 2.8))
        g.plot(kind="barh", ax=ax, color="#ff7f0e")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.subheader("Experiment groups")
if len(df_f):
    exp = df_f.groupby("experiment_group", observed=True).agg(
        sessions=("session_id", "count"),
        purchase_rate=("has_purchase", "mean"),
        bounce_rate=("has_bounce", "mean"),
        cart_rate=("has_add_to_cart", "mean"),
        median_duration=("duration_sec", "median"),
    )
    st.dataframe(exp.round(4))
    if PLOTLY_AVAILABLE:
        plot_df = exp.reset_index()
        fig = px.bar(plot_df, x="experiment_group", y="purchase_rate", text="purchase_rate", height=320)
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        st.plotly_chart(fig, width="stretch")

# ---------------- A/B статтест (кратко) ----------------
st.markdown("---")
st.subheader("A/B test (preview)")
st.caption("Полный дашборд: `streamlit run scripts/EDA_ab_test.py`")

if len(df_f) and df_f["experiment_group"].nunique() >= 2:
    summary_ab, overall_ab, pairwise_ab = run_ab_analysis(df_f)
    srm_row = overall_ab.loc[overall_ab["test"] == "SRM_chisquare"].iloc[0]
    purch = pairwise_ab.loc[pairwise_ab["metric"] == "has_purchase"].copy()

    k1, k2, k3 = st.columns(3)
    k1.metric("SRM ok?", "yes" if not srm_row["significant_alpha_05"] else "NO")
    k2.metric("SRM p-value", f"{srm_row['p_value']:.3g}")
    best = summary_ab["has_purchase_rate"].idxmax()
    k3.metric("Best purchase group", f"{best} ({summary_ab.loc[best, 'has_purchase_rate']*100:.2f}%)")

    st.dataframe(summary_ab[["sessions", "has_purchase_rate", "has_add_to_cart_rate", "has_bounce_rate"]].round(4))

    if len(purch):
        st.markdown("**Purchase pairwise (Bonferroni)**")
        for _, r in purch.iterrows():
            mark = "значимо" if r["significant_bonferroni"] else "не значимо"
            st.write(
                f"- {r['group_a']} -> {r['group_b']}: "
                f"delta {r['diff_pp']:+.2f} pp "
                f"(CI [{r['ci95_lo_pp']:.2f}; {r['ci95_hi_pp']:.2f}]) — {mark}"
            )
else:
    st.info("Нужно >= 2 experiment_group в фильтре")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Campaign channel vs CR")
    if len(df_f) and "campaign_channel" in df_f.columns:
        g = (
            df_f.groupby("campaign_channel", dropna=False)
            .agg(sessions=("session_id", "count"), cr=("has_purchase", "mean"))
            .sort_values("cr", ascending=False)
        )
        st.dataframe(g.round(4))
with c2:
    st.subheader("Duration vs conversion (sample)")
    sample = df_f.sample(n=min(30_000, len(df_f)), random_state=42) if len(df_f) else df_f
    if len(sample) and SEABORN_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.boxplot(
            x=sample["has_purchase"].map({0: "no purchase", 1: "purchase"}),
            y=sample["duration_sec"],
            ax=ax,
            showfliers=False,
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.subheader("Preview (sessions)")
show = [
    c for c in [
        "session_id", "session_start", "customer_id", "device_type", "traffic_source",
        "experiment_group", "campaign_id", "campaign_channel", "n_events", "duration_sec",
        "has_view", "has_click", "has_add_to_cart", "has_purchase", "has_bounce", "funnel_step",
    ] if c in df_f.columns
]
st.dataframe(arrow_safe(df_f[show].head(300)).reset_index(drop=True))
st.caption(f"Sessions from events — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
