# EDA_ab_test.py
# Что делает: Streamlit-дашборд A/B статтеста по experiment_group (сессии).
# SRM, chi2, pairwise z-test, 95% CI, lift, Bonferroni. 


from pathlib import Path
import datetime
import sys
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_test_sessions import run_ab_analysis, ALPHA  # noqa: E402

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="EDA — A/B test", layout="wide")

SRC = Path("data") / "snapshot" / "sessions_snap.parquet"
METRICS = ["has_purchase", "has_add_to_cart", "has_bounce"]
METRIC_LABELS = {
    "has_purchase": "Purchase CR",
    "has_add_to_cart": "Add to cart rate",
    "has_bounce": "Bounce rate",
}


@st.cache_data
def load_sessions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "session_start" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["session_start"]):
        df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
    return df


def cats(series):
    try:
        return list(series.cat.categories)
    except Exception:
        return sorted([x for x in series.dropna().unique().tolist()])


if not SRC.exists():
    st.error(f"Нет {SRC}. Сначала build_sessions.py")
    st.stop()

df = load_sessions(SRC)

# ----------------- Sidebar -----------------
st.sidebar.header("Фильтры")
min_d, max_d = df["session_start"].min().date(), df["session_start"].max().date()
dr = st.sidebar.date_input("Период session_start", value=(min_d, max_d))
start_d, end_d = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dr, dr))

sel_device = st.sidebar.multiselect("Device", cats(df["device_type"]), default=cats(df["device_type"]))
sel_src = st.sidebar.multiselect("Traffic", cats(df["traffic_source"]), default=cats(df["traffic_source"]))
metric = st.sidebar.selectbox(
    "Основная метрика",
    options=METRICS,
    format_func=lambda m: METRIC_LABELS[m],
    index=0,
)
baseline = st.sidebar.selectbox(
    "Baseline для lift",
    options=["Control", "Variant_A", "Variant_B"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(f"alpha = {ALPHA} · Bonferroni на pairwise · N большой → смотри Δ pp")

mask = (df["session_start"].dt.date >= start_d) & (df["session_start"].dt.date <= end_d)
if sel_device:
    mask &= df["device_type"].isin(sel_device)
if sel_src:
    mask &= df["traffic_source"].isin(sel_src)
df_f = df.loc[mask].copy()

if df_f["experiment_group"].nunique() < 2:
    st.warning("В фильтре меньше 2 групп — ослабь фильтры")
    st.stop()

summary, overall, pairwise = run_ab_analysis(df_f, metrics=METRICS)

# ----------------- Header / KPIs -----------------
st.title("A/B Test Dashboard")
st.markdown("Статтест на **сессиях**: SRM · chi² · z-test долей · 95% CI · Bonferroni")

srm = overall.loc[overall["test"] == "SRM_chisquare"].iloc[0]
chi = overall.loc[(overall["test"] == "chi2_independence") & (overall["metric"] == metric)]
chi_p = float(chi.iloc[0]["p_value"]) if len(chi) else np.nan

# лучший вариант vs baseline по выбранной метрике
rates = summary[f"{metric}_rate"]
base_rate = float(rates.get(baseline, np.nan)) if baseline in rates.index else float(rates.iloc[0])
best_g = rates.idxmax() if metric != "has_bounce" else rates.idxmin()
best_rate = float(rates.loc[best_g])
best_lift_pp = (best_rate - base_rate) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sessions (filtered)", f"{len(df_f):,}")
k2.metric("SRM p-value", f"{srm['p_value']:.3g}")
k3.metric("SRM ok?", "yes" if not srm["significant_alpha_05"] else "NO")
k4.metric(f"Chi² {METRIC_LABELS[metric]}", f"{chi_p:.3g}")
k5.metric(f"Best vs {baseline}", f"{best_g} ({best_lift_pp:+.2f} pp)")

st.markdown("---")

# ----------------- Rates + traffic -----------------
left, right = st.columns((1.2, 1))

with left:
    st.subheader("Rates by experiment group")
    plot_sum = summary.reset_index().rename(columns={"index": "experiment_group"})
    if "experiment_group" not in plot_sum.columns:
        plot_sum = summary.copy()
        plot_sum["experiment_group"] = plot_sum.index.astype(str)
        plot_sum = plot_sum.reset_index(drop=True)

    rate_col = f"{metric}_rate"
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            plot_sum,
            x="experiment_group",
            y=rate_col,
            text=rate_col,
            color="experiment_group",
            height=360,
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".1%", showlegend=False, yaxis_title=METRIC_LABELS[metric])
        st.plotly_chart(fig, width="stretch")
    else:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        plot_sum.set_index("experiment_group")[rate_col].plot(kind="bar", ax=ax, color="#636EFA")
        ax.set_ylabel(METRIC_LABELS[metric])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with right:
    st.subheader("Traffic share (SRM)")
    shares = (summary["sessions"] / summary["sessions"].sum()).reset_index()
    shares.columns = ["experiment_group", "share"]
    if PLOTLY_AVAILABLE:
        fig = px.pie(shares, names="experiment_group", values="share", hole=0.35, height=360)
        st.plotly_chart(fig, width="stretch")
    else:
        st.dataframe(shares.assign(share_pct=(shares["share"] * 100).round(2)))
    st.caption(str(srm.get("note", "")))

st.markdown("**Summary table**")
st.dataframe(summary.round(4))

st.markdown("---")

# ----------------- Pairwise -----------------
st.subheader(f"Pairwise z-tests — {METRIC_LABELS[metric]}")
pw = pairwise.loc[pairwise["metric"] == metric].copy()
# удобный порядок: сначала сравнения с baseline
pw["vs_baseline"] = (pw["group_a"] == baseline) | (pw["group_b"] == baseline)
pw = pw.sort_values(["vs_baseline", "p_value"], ascending=[False, True])

pcols = [
    "group_a", "group_b", "p1", "p2", "diff_pp", "lift_rel",
    "ci95_lo_pp", "ci95_hi_pp", "z", "p_value", "p_bonferroni",
    "significant_raw", "significant_bonferroni", "n1", "n2",
]
st.dataframe(pw[[c for c in pcols if c in pw.columns]].round(4))

c1, c2 = st.columns(2)

with c1:
    st.subheader("Lift (pp) + 95% CI")
    if len(pw) and PLOTLY_AVAILABLE:
        plot_pw = pw.copy()
        plot_pw["pair"] = plot_pw["group_a"].astype(str) + " -> " + plot_pw["group_b"].astype(str)
        fig = go.Figure()
        for _, r in plot_pw.iterrows():
            color = "#2ca02c" if r["significant_bonferroni"] else "#aaaaaa"
            fig.add_trace(
                go.Scatter(
                    x=[r["ci95_lo_pp"], r["ci95_hi_pp"]],
                    y=[r["pair"], r["pair"]],
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[r["diff_pp"]],
                    y=[r["pair"]],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    name=r["pair"],
                    showlegend=False,
                    hovertemplate=(
                        "%{y}<br>delta=%{x:.2f} pp"
                        f"<br>p_bonf={r['p_bonferroni']:.3g}<extra></extra>"
                    ),
                )
            )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=340,
            xaxis_title="delta pp (group_b - group_a)",
            yaxis_title="",
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Зелёный = значимо после Bonferroni; серый = нет")
    elif len(pw):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        y = np.arange(len(pw))
        ax.hlines(y, pw["ci95_lo_pp"], pw["ci95_hi_pp"], color="#636EFA")
        ax.plot(pw["diff_pp"], y, "o", color="#1f77b4")
        ax.axvline(0, color="gray", ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(pw["group_a"].astype(str) + " -> " + pw["group_b"].astype(str))
        ax.set_xlabel("delta pp")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with c2:
    st.subheader("Overall tests")
    ocols = [c for c in ["test", "metric", "chi2", "p_value", "significant_alpha_05", "note"] if c in overall.columns]
    st.dataframe(overall[ocols])

    st.subheader("Вердикт")
    for _, r in pw.iterrows():
        mark = "значимо" if r["significant_bonferroni"] else "не значимо"
        st.write(
            f"- **{r['group_a']} -> {r['group_b']}**: "
            f"{r['p1']*100:.2f}% -> {r['p2']*100:.2f}% "
            f"(delta {r['diff_pp']:+.2f} pp, "
            f"95% CI [{r['ci95_lo_pp']:.2f}; {r['ci95_hi_pp']:.2f}], "
            f"Bonferroni: {mark})"
        )

st.markdown("---")
st.info(
    "N очень большой — даже маленький lift легко становится statistically significant. "
    "Для решения смотри **delta pp**, CI и бизнес-порог, не только p-value. "
    "Полный прогон в терминал: `python scripts/ab_test_sessions.py`."
)

st.caption(f"A/B dashboard · {SRC} · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
