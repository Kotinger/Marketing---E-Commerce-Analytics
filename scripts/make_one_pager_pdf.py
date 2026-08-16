# make_one_pager_pdf.py
# Что делает: собирает docs/ONE_PAGER.pdf (A4) из ключевых тезисов проекта.

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

out = Path("docs/ONE_PAGER.pdf")
fig = plt.figure(figsize=(8.27, 11.69))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.07, 0.05, 0.86, 0.90])
ax.axis("off")

# text, size, weight, style
lines = [
    ("E-commerce Analytics — One-pager", 16, "bold", "normal"),
    ("Pet-project: QA → dim/fact prep → sales join → session funnel", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("ДАННЫЕ", 11, "bold", "normal"),
    ("campaigns 50 · customers 100k · products 2k · transactions 103k", 9, "normal", "normal"),
    ("events 2M → sessions 633k  |  keys: customer_id, product_id, campaign_id (0=none)", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("ПОДХОД", 11, "bold", "normal"),
    ("1) QA  2) Dim prep + dtypes  3) Fact prep + sessionization", 9, "normal", "normal"),
    ("4) EDA dims (campaigns/customers)  5) EDA sales (joins)  6) EDA behavior (funnel)", 9, "normal", "normal"),
    ("products — только dim; events — не сырой EDA, а сессии", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("НАХОДКИ — КАЧЕСТВО", 11, "bold", "normal"),
    ("~10.1% tx без product/revenue · refund ≈ 2.9% · traffic_source нормализован", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("НАХОДКИ — SALES", 11, "bold", "normal"),
    ("Топ выручки: Electronics → Home → Fashion", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("НАХОДКИ — BEHAVIOR + A/B", 11, "bold", "normal"),
    ("Session CR ≈ 15.1% · bounce ≈ 26%", 9, "normal", "normal"),
    ("CR выше: Email / Paid Search / Social (~17–18%); Organic/Direct (~12%)", 9, "normal", "normal"),
    ("A/B 60/20/20, SRM ok · Purchase: Control 14.72% → B +1.35 pp (Bonferroni sig)", 9, "normal", "normal"),
    ("Variant_A +0.48 pp vs Control · cart без эффекта · смотрим Δ pp, не только p", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("АРТЕФАКТЫ", 11, "bold", "normal"),
    ("scripts/load&transform_* · dtypes_* · build_sessions.py · ab_test_sessions.py", 9, "normal", "normal"),
    ("EDA_campaigns · EDA_customers · EDA_sales · EDA_behavior (Streamlit)", 9, "normal", "normal"),
    ("Стек: Python, pandas, Parquet, Streamlit, matplotlib/seaborn/plotly", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("ВЫВОД", 11, "bold", "normal"),
    ("Правильный уровень агрегации + join’ы → ясные деньги и конверсия;", 9, "normal", "normal"),
    ("сырой event-log для EDA не тащим.", 9, "normal", "normal"),
    ("", 8, "normal", "normal"),
    ("Подробности и запуск — README.md", 8, "normal", "italic"),
]

y = 0.98
for text, size, weight, style in lines:
    if text == "":
        y -= 0.015
        continue
    ax.text(
        0.0,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        va="top",
        ha="left",
        family="DejaVu Sans",
        transform=ax.transAxes,
    )
    y -= 0.030 if size >= 11 else 0.023

with PdfPages(out) as pdf:
    pdf.savefig(fig, dpi=150)
plt.close(fig)
print("Saved:", out)
