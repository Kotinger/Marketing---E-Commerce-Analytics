# ab_test_sessions.py
# Что делает: статтест A/B по experiment_group на уровне сессий.
# Метрики: purchase / add_to_cart / bounce. Считаю chi2, z-test пропорций, CI, lift, SRM.
# Результат печатаю и сохраняю в data/snapshot/ab_test_results.csv.

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats

SRC = Path("data/snapshot") / "sessions_snap.parquet"
OUT = Path("data/snapshot") / "ab_test_results.csv"
ALPHA = 0.05


def load_sessions(path: Path = SRC) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Сначала build_sessions.py")
    return pd.read_parquet(path)


def srm_check(counts: pd.Series, expected_share: dict | None = None) -> dict:
    """Sample Ratio Mismatch.
    По умолчанию жду 60/20/20 (Control / A / B) — так выглядит факт в данных.
    Если доли заданы иначе — передай expected_share."""
    counts = counts.astype(float)
    groups = list(counts.index)
    if expected_share is None:
        # эвристика: если Control заметно больше — 60/20/20, иначе equal
        if "Control" in groups and counts.get("Control", 0) > counts.sum() * 0.4:
            expected_share = {g: (0.6 if g == "Control" else 0.2) for g in groups}
            # нормализуем на всякий случай
            s = sum(expected_share[g] for g in groups)
            expected_share = {g: expected_share[g] / s for g in groups}
        else:
            expected_share = {g: 1 / len(groups) for g in groups}

    observed = np.array([counts[g] for g in groups], dtype=float)
    expected = np.array([expected_share[g] * observed.sum() for g in groups], dtype=float)
    chi2, p = stats.chisquare(observed, expected)
    return {
        "test": "SRM_chisquare",
        "groups": ", ".join(groups),
        "metric": "traffic_share",
        "chi2": chi2,
        "p_value": p,
        "significant_alpha_05": p < ALPHA,
        "note": "expected=" + ", ".join(f"{g}:{expected_share[g]:.0%}" for g in groups),
    }


def proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Двусторонний z-test разности долей + 95% CI на разность (p2 - p1)."""
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se_pool == 0:
        z, p = 0.0, 1.0
    else:
        z = (p2 - p1) / se_pool
        p = 2 * stats.norm.sf(abs(z))

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = stats.norm.ppf(1 - ALPHA / 2)
    diff = p2 - p1
    ci_lo = diff - zcrit * se_diff
    ci_hi = diff + zcrit * se_diff

    return {
        "p1": p1,
        "p2": p2,
        "diff_pp": diff * 100,  # percentage points
        "lift_rel": (diff / p1) if p1 > 0 else np.nan,
        "z": z,
        "p_value": p,
        "ci95_lo_pp": ci_lo * 100,
        "ci95_hi_pp": ci_hi * 100,
        "n1": n1,
        "n2": n2,
        "x1": x1,
        "x2": x2,
    }


def chi2_independence(df: pd.DataFrame, group_col: str, metric_col: str) -> dict:
    """Общий chi2: experiment_group × metric (0/1)."""
    ct = pd.crosstab(df[group_col], df[metric_col])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {
        "test": "chi2_independence",
        "metric": metric_col,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "significant_alpha_05": p < ALPHA,
    }


def pairwise_tests(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """Все пары групп + Bonferroni на число сравнений."""
    if groups is None:
        groups = sorted(df[group_col].dropna().unique().tolist())
    pairs = list(itertools.combinations(groups, 2))
    rows = []
    for a, b in pairs:
        s1 = df.loc[df[group_col] == a, metric_col].astype(int)
        s2 = df.loc[df[group_col] == b, metric_col].astype(int)
        r = proportion_ztest(int(s1.sum()), int(len(s1)), int(s2.sum()), int(len(s2)))
        r.update(
            {
                "test": "z_proportion",
                "metric": metric_col,
                "group_a": a,
                "group_b": b,
                "baseline": a,
                "variant": b,
            }
        )
        rows.append(r)

    out = pd.DataFrame(rows)
    m = max(len(out), 1)
    out["p_bonferroni"] = (out["p_value"] * m).clip(upper=1.0)
    out["significant_raw"] = out["p_value"] < ALPHA
    out["significant_bonferroni"] = out["p_bonferroni"] < ALPHA
    return out


def summary_by_group(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    g = df.groupby("experiment_group", observed=True)
    rows = {"sessions": g.size()}
    for m in metrics:
        rows[f"{m}_rate"] = g[m].mean()
        rows[f"{m}_count"] = g[m].sum()
    return pd.DataFrame(rows)


def run_ab_analysis(df: pd.DataFrame, metrics: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Главная функция: summary + overall chi2 + pairwise z-tests."""
    if metrics is None:
        metrics = ["has_purchase", "has_add_to_cart", "has_bounce"]

    work = df.dropna(subset=["experiment_group"]).copy()
    for m in metrics:
        work[m] = work[m].astype(int)

    summary = summary_by_group(work, metrics)

    """SRM по факту сплита."""
    counts = work["experiment_group"].value_counts()
    srm = srm_check(counts)

    overall_rows = [srm]
    pair_parts = []
    for m in metrics:
        overall_rows.append(chi2_independence(work, "experiment_group", m))
        pair_parts.append(pairwise_tests(work, "experiment_group", m))

    overall = pd.DataFrame(overall_rows)
    pairwise = pd.concat(pair_parts, ignore_index=True)
    return summary, overall, pairwise


def main():
    print("Гружу sessions...")
    df = load_sessions()
    print("sessions:", len(df))

    """Смотрю доли групп — Control больше, похоже на 60/20/20."""
    print("\nДоли experiment_group:")
    print(df["experiment_group"].value_counts(normalize=True).round(4))

    summary, overall, pairwise = run_ab_analysis(df)

    print("\n=== Summary rates ===")
    print(summary.round(4))

    print("\n=== Overall / SRM ===")
    cols_o = [c for c in ["test", "metric", "chi2", "p_value", "significant_alpha_05", "note"] if c in overall.columns]
    print(overall[cols_o].to_string(index=False))

    print("\n=== Pairwise z-tests (purchase focus first) ===")
    show = pairwise.copy()
    show = show.sort_values(["metric", "p_value"])
    cols = [
        "metric", "group_a", "group_b", "p1", "p2", "diff_pp", "lift_rel",
        "ci95_lo_pp", "ci95_hi_pp", "p_value", "p_bonferroni",
        "significant_raw", "significant_bonferroni", "n1", "n2",
    ]
    print(show[cols].round(4).to_string(index=False))

    """Сохраняю pairwise — это основная таблица для портфолио."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    show[cols].to_csv(OUT, index=False)
    print("\nSaved:", OUT)

    """Короткий вывод словами."""
    purch = show[show["metric"] == "has_purchase"].copy()
    print("\n=== Короткий вывод (purchase) ===")
    for _, r in purch.iterrows():
        sig = "да" if r["significant_bonferroni"] else "нет"
        print(
            f"{r['group_a']} -> {r['group_b']}: "
            f"{r['p1']*100:.2f}% -> {r['p2']*100:.2f}% "
            f"(delta {r['diff_pp']:+.2f} pp, rel lift {r['lift_rel']*100:+.1f}%, "
            f"95% CI [{r['ci95_lo_pp']:.2f}; {r['ci95_hi_pp']:.2f}] pp, "
            f"Bonferroni sig={sig})"
        )
    print(
        "\nВажно: N очень большой - даже маленький lift легко значим. "
        "Смотрим ещё на delta pp и CI, не только на p-value."
    )


if __name__ == "__main__":
    main()
