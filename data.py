import numpy as np
import pandas as pd


def _make_df(name: str, y: np.ndarray, start: str = "2019-01-01") -> pd.DataFrame:
    n = len(y)
    return pd.DataFrame(
        {
            "unique_id": [name] * n,
            "ds": pd.date_range(start, periods=n, freq="MS"),
            "y": y.clip(min=0).round(2),
        }
    )


def retail_demand(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Strong annual seasonality + stable trend. Statistical models should win."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 200 + 1.2 * t
    seasonality = 60 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 8, n)
    return _make_df("retail_demand", trend + seasonality + noise)


def cash_inflow(n: int = 60, seed: int = 7, name: str = "cash_inflow") -> pd.DataFrame:
    """High volatility, weak seasonality. ML lag models should win."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 1000 + 0.8 * t
    seasonality = 15 * np.sin(2 * np.pi * t / 12)
    volatility = rng.normal(0, 180, n)
    regime = np.where(t > 40, 120, 0)
    return _make_df(name, trend + seasonality + volatility + regime)


def support_volume(n: int = 60, seed: int = 99) -> pd.DataFrame:
    """Same profile as cash_inflow — different seed. Used to show memory retrieval."""
    return cash_inflow(n=n, seed=seed, name="support_volume")


DEMO_SERIES = {
    "Retail Demand (Seasonal)": retail_demand(),
    "Cash Inflow (Volatile)": cash_inflow(),
    "Support Volume (Volatile)": support_volume(),
}

if __name__ == "__main__":
    for name, df in DEMO_SERIES.items():
        print(f"\n{name}: {len(df)} rows | y range [{df['y'].min():.0f}, {df['y'].max():.0f}]")
        print(df.tail(3).to_string(index=False))
