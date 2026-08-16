# E-commerce Analytics Pet Project

Пет-проект по анализу данных e-commerce. 
Датасет : https://www.kaggle.com/datasets/geethasagarbonthu/marketing-and-e-commerce-analytics-dataset

Интерактивные дашборды на **Streamlit**/

---

## О чём проект

Есть 5 связанных таблиц (кампании, клиенты, товары, транзакции, события).  
Цель — пройти полный цикл аналитика: качество данных → подготовка → EDA → продажи → воронка.
+ Интерактивные дашборды на **Streamlit**/

---

## Данные

| Таблица | Строк | Роль |
|---|---:|---|
| `campaigns` | 50 | справочник кампаний |
| `customers` | 100 000 | справочник клиентов |
| `products` | 2 000 | справочник товаров |
| `transactions` | 103 127 | факты продаж |
| `events` | ~2 000 000 | сырые события → сессии |

Связи: `customer_id`, `product_id`, `campaign_id` (`0` = без кампании).

---

## Ключевые находки

**Качество**
- В transactions ~**10.1%** строк без `product_id` и `gross_revenue` (одни и те же кейсы).
- Refund rate ≈ **2.9%** (отрицательный revenue).
- В events `traffic_source` был в разном регистре → нормализован.

**Sales**
- Основная выручка по категориям: **Electronics → Home → Fashion**.
- Часть продаж без кампании (`campaign_id = 0`).

**Behavior (633 462 сессии)**
- Session CR (purchase) ≈ **15.1%**, bounce-сессии ≈ **26%**.
- CR выше у **Email / Paid Search / Social** (~17–18%), ниже у Organic/Direct (~12%).
- Device почти не различается (~15%).

**A/B (статтест на сессиях)**
- Сплит **60/20/20** (Control / A / B), SRM ok (p ≈ 0.51).
- Purchase: Control **14.72%** → A **15.20%** (+0.48 pp) → B **16.06%** (+1.35 pp vs Control).
- Все пары по purchase значимы после Bonferroni; cart — нет; bounce — слабо / не держится после поправки.
- N огромный → смотрим ещё **Δ pp и CI**, не только p-value (`scripts/ab_test_sessions.py`).

---

## Pipeline

```text
Исходные данные (папка data/)
   └── *.csv  (все входные файлы)

Шаг 1. Загрузка и первичная трансформация
   ├── load&transform_*.py  →  для каждого CSV создаётся снапшот:
   │     • данные/снапшот/*_snap.csv  (кроме events – для него паркет)
   │     • данные/снапшот/events_snap.parquet
   ├── dtypes_*.py  →  приводит типы колонок и сохраняет результат как:
   │     • данные/снапшот/*_snap.parquet  (все снапшоты уже в паркете)
   └── build_sessions.py  →  строит сессионный снапшот:
         • данные/снапшот/sessions_snap.parquet

Шаг 2. Ветвление на исследовательские задачи (EDA)
   Из папки data/snapshot/ берутся подготовленные паркеты для трёх направлений:
   ├── EDA_campaigns    – анализ кампаний
   ├── EDA_customers    – анализ клиентов
   └── EDA_sales        – анализ продаж
   
Шаг 3. Ветвление на исследовательские задачи (EDA)
   Из папки data/snapshot/ берутся подготовленные паркеты
   DASH_behavior (на основе sessions_funnel) – анализ воронки поведения
   DASH_ab_test по experiment_group (сессии) - статтест A/B 
```

### Скрипты

**Загрузка и обработка**
| Файл | Что делает |
|---|---|
| `scripts/load&transform_campaings.py` | campaigns: длина, сезон, категория длины → snap |
| `scripts/load&transform_customers.py` | customers: age/tenure → snap |
| `scripts/load&transform_products.py` | products: price_tier / season (только dim-prep) |
| `scripts/load&transform_transactions.py` | transactions: флаги, время, discount bins |
| `scripts/load&transform_events.py` | events: нормализация, флаги → parquet + sample |
| `scripts/dtypes_*.py` | типы данных → parquet |
| `scripts/build_sessions.py` | events → session-level воронка |
| `scripts/ab_test_sessions.py` | A/B статтест (chi², z-test, CI, SRM, Bonferroni) |
| `scripts/make_one_pager_pdf.py` | пересобрать `docs/ONE_PAGER.pdf` |

**Дашборды**
| Файл | Что делает |
|---|---|
| `scripts/EDA_campaigns.py` | профиль кампаний |
| `scripts/EDA_customers.py` | профиль клиентов |
| `scripts/EDA_sales.py` | продажи: tx ⋈ customers ⋈ products ⋈ campaigns |
| `scripts/EDA_behavior.py` | воронка / traffic / device (+ краткий A/B preview) |
| `scripts/EDA_ab_test.py` | полный A/B дашборд: SRM, chi², z-test, CI, Bonferroni |

Краткий one-pager: [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md) · PDF: [`docs/ONE_PAGER.pdf`](docs/ONE_PAGER.pdf)  
Опора для слайдов: [`docs/SLIDES_OUTLINE.md`](docs/SLIDES_OUTLINE.md)

---

## Быстрый старт

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Прогнать pipeline (если нет snapshot’ов)

```bash
python "scripts/load&transform_campaings.py"
python scripts/dtypes_campaigns.py

python "scripts/load&transform_customers.py"
python scripts/dtypes_customers.py

python "scripts/load&transform_products.py"
python scripts/dtypes_products.py

python "scripts/load&transform_transactions.py"
python scripts/dtypes_transactions.py

# нужен data/events.csv локально
python "scripts/load&transform_events.py"
python scripts/dtypes_events.py
python scripts/build_sessions.py
python scripts/ab_test_sessions.py
```

### Дашборды

```bash
streamlit run scripts/EDA_campaigns.py
streamlit run scripts/EDA_customers.py
streamlit run scripts/EDA_sales.py
streamlit run scripts/EDA_behavior.py
streamlit run scripts/EDA_ab_test.py
```

A/B: терминал `python scripts/ab_test_sessions.py` или дашборд `EDA_ab_test.py`.

Запускай из корня репозитория (пути вида `data/snapshot/...`).

---

## Стек

Python, pandas, pyarrow, Streamlit, matplotlib, seaborn, plotly.

---

## Структура репозитория

```text
├── data/
│   ├── *.csv                 # исходники (events.csv — локально)
│   └── snapshot/             # обработанные parquet/csv
├── scripts/                  # pipeline + EDA
├── docs/
│   ├── ONE_PAGER.md
│   └── SLIDES_OUTLINE.md
├── requirements.txt
└── README.md
```

---

## Логика выбора анализа

- **products** — без отдельного тяжёлого EDA: товар раскрывается в `EDA_sales`.
- **events** — не value_counts на 2M строк, а `build_sessions` + воронка в `EDA_behavior`.
- **transactions** — главный денежный слой только после join с dims.

---

## Автор

Pet-project по анализу данных (portfolio).  
Если будешь форкать — поставь ⭐ и опиши свой датасет в issue/PR.
