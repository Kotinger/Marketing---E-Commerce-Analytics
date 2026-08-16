# build_sessions.py
# Что делает: из events_snap.parquet собирает session-level таблицу (воронка/флаги/device/traffic/experiment)
# и сохраняет sessions_snap.parquet. Сырые 2M событий для EDA не гружу — смотрю сессии

from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("data/snapshot") / "events_snap.parquet"
OUT = Path("data/snapshot") / "sessions_snap.parquet"

if not SRC.exists():
    print(f"Нет {SRC}. Сначала load&transform_events.py и dtypes_events.py")
    raise SystemExit(1)

df = pd.read_parquet(SRC)
print("events:", df.shape)

"""Проверяю timestamp и раскладываю event_type в флаги view/click/cart/purchase/bounce."""
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

event_flags = pd.get_dummies(df["event_type"], dtype="int8")
for col in ["view", "click", "add_to_cart", "purchase", "bounce"]:
    if col not in event_flags.columns:
        event_flags[col] = 0

work = pd.concat(
    [
        df[
            [
                "session_id",
                "customer_id",
                "timestamp",
                "device_type",
                "traffic_source",
                "campaign_id",
                "page_category",
                "session_duration_sec",
                "experiment_group",
            ]
        ],
        event_flags[["view", "click", "add_to_cart", "purchase", "bounce"]],
    ],
    axis=1,
)

"""Агрегирую до 1 строки = 1 session_id.
campaign_id беру max: если хоть раз был attributed event — сессия с кампанией."""
agg = work.groupby("session_id", sort=False).agg(
    customer_id=("customer_id", "first"),
    session_start=("timestamp", "min"),
    session_end=("timestamp", "max"),
    n_events=("timestamp", "size"),
    device_type=("device_type", "first"),
    traffic_source=("traffic_source", "first"),
    experiment_group=("experiment_group", "first"),
    campaign_id=("campaign_id", "max"),
    duration_sec=("session_duration_sec", "max"),
    n_view=("view", "sum"),
    n_click=("click", "sum"),
    n_add_to_cart=("add_to_cart", "sum"),
    n_purchase=("purchase", "sum"),
    n_bounce=("bounce", "sum"),
)

sessions = agg.reset_index()
sessions["has_view"] = (sessions["n_view"] > 0).astype("int8")
sessions["has_click"] = (sessions["n_click"] > 0).astype("int8")
sessions["has_add_to_cart"] = (sessions["n_add_to_cart"] > 0).astype("int8")
sessions["has_purchase"] = (sessions["n_purchase"] > 0).astype("int8")
sessions["has_bounce"] = (sessions["n_bounce"] > 0).astype("int8")
sessions["has_campaign"] = (sessions["campaign_id"] > 0).astype("int8")
sessions["converted"] = sessions["has_purchase"]

"""Максимальный шаг воронки в сессии (не обязательно строго линейный путь)."""
sessions["funnel_step"] = np.select(
    [
        sessions["has_purchase"] == 1,
        sessions["has_add_to_cart"] == 1,
        sessions["has_click"] == 1,
        sessions["has_view"] == 1,
        sessions["has_bounce"] == 1,
    ],
    ["purchase", "add_to_cart", "click", "view", "bounce"],
    default="other",
)

sessions["session_date"] = sessions["session_start"].dt.date.astype(str)
sessions["session_month"] = sessions["session_start"].dt.to_period("M").astype(str)

"""Категории + компактные типы — как в dtypes_*."""
for c in [
    "device_type",
    "traffic_source",
    "experiment_group",
    "funnel_step",
    "session_month",
    "session_date",
]:
    sessions[c] = sessions[c].astype("category")

sessions["campaign_id"] = sessions["campaign_id"].astype("Int8")
sessions["customer_id"] = sessions["customer_id"].astype("Int32")
sessions["n_events"] = sessions["n_events"].astype("Int16")
sessions["duration_sec"] = sessions["duration_sec"].astype("float32")

sessions.to_parquet(OUT, index=False)
print("Saved:", OUT, sessions.shape)
print(sessions[["has_view", "has_click", "has_add_to_cart", "has_purchase", "has_bounce"]].mean().round(4))
print("funnel_step:\n", sessions["funnel_step"].value_counts())
