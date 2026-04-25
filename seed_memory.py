"""
Run this once before the demo to pre-populate memory with past forecasting lessons.
Usage: uv run python seed_memory.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from agents.profiler import profile_series
from agents.tournament import run_tournament
from agents.memory import save_lesson, _load, MEMORY_FILE
from agents.reflection import generate_lesson


def _make_df(name, y, start="2018-01-01"):
    n = len(y)
    return pd.DataFrame({
        "unique_id": [name] * n,
        "ds": pd.date_range(start, periods=n, freq="MS"),
        "y": np.clip(y, 0, None).round(2),
    })


PAST_SERIES = [
    # --- Volatile, weak seasonality (will match Cash Inflow / Support Volume) ---
    {
        "name": "collections_volume",
        "label": "Collections Volume (Finance)",
        "fn": lambda: _make_df("collections_volume",
            500 + 0.5 * np.arange(60)
            + 12 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(11).normal(0, 95, 60)
            + np.where(np.arange(60) > 45, 80, 0)),
    },
    {
        "name": "cloud_spend",
        "label": "Cloud Infrastructure Spend",
        "fn": lambda: _make_df("cloud_spend",
            800 + 2 * np.arange(60)
            + 8 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(22).normal(0, 130, 60)),
    },
    {
        "name": "enterprise_tickets",
        "label": "Enterprise Support Tickets",
        "fn": lambda: _make_df("enterprise_tickets",
            300 + 0.4 * np.arange(60)
            + 10 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(33).normal(0, 70, 60)
            + np.where(np.arange(60) > 50, 60, 0)),
    },
    # --- Strong seasonality, stable (will match Retail Demand) ---
    {
        "name": "ecommerce_orders",
        "label": "E-commerce Orders",
        "fn": lambda: _make_df("ecommerce_orders",
            1000 + 1.5 * np.arange(60)
            + 200 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(44).normal(0, 25, 60)),
    },
    {
        "name": "seasonal_ad_spend",
        "label": "Seasonal Ad Spend",
        "fn": lambda: _make_df("seasonal_ad_spend",
            400 + 0.8 * np.arange(60)
            + 80 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(55).normal(0, 15, 60)),
    },
    # --- Mixed ---
    {
        "name": "saas_arr",
        "label": "SaaS Annual Recurring Revenue",
        "fn": lambda: _make_df("saas_arr",
            2000 + 8 * np.arange(60)
            + 30 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.default_rng(66).normal(0, 60, 60)),
    },
]


def main():
    # Clear existing memory for a clean seed
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
        print("Cleared existing memory.")

    print(f"Seeding memory with {len(PAST_SERIES)} past forecasting runs...\n")

    for entry in PAST_SERIES:
        df = entry["fn"]()
        label = entry["label"]
        print(f"  [{label}]")

        profile = profile_series(df)
        vol = "high" if profile["volatility"] > 0.15 else "low"
        seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"
        print(f"    Profile: {vol} volatility · {seas} seasonality")

        results, winner = run_tournament(df, horizon=3)
        wapes = {k: f"{v['wape']:.1%}" for k, v in results.items()}
        print(f"    Winner: {winner} | WAPEs: {wapes}")

        lesson = generate_lesson(profile, results, winner, series_label=label)
        save_lesson(lesson)
        print(f"    Lesson: {lesson['lesson_text'][:90]}...")
        print()

    lessons = _load()
    print(f"Memory seeded with {len(lessons)} lessons.")
    print("\nVoilà — run the demo now:")
    print("  uv run streamlit run app.py")
    print("\nDemo order for best story:")
    print("  1. Cash Inflow (Volatile)    → 3 similar memories retrieved → GradientBoosting recommended")
    print("  2. Support Volume (Volatile) → memory reinforced")
    print("  3. Retail Demand (Seasonal)  → different memory cluster → ETS recommended")


if __name__ == "__main__":
    main()
