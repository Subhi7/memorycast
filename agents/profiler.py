import numpy as np
import pandas as pd


def profile_series(df: pd.DataFrame, freq: str = "MS") -> dict:
    """Return a flat profile dict for a time series DataFrame with columns [ds, y]."""
    y = df["y"].values.astype(float)
    n = len(y)
    season_lag = 12  # monthly assumed

    # Trend strength: R² of linear fit
    x = np.arange(n)
    coeffs = np.polyfit(x, y, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    trend_strength = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Seasonality strength: autocorrelation at annual lag
    if n > season_lag * 2:
        seasonality_strength = float(max(0.0, pd.Series(y).autocorr(lag=season_lag)))
    else:
        seasonality_strength = 0.0

    # Volatility: coefficient of variation
    volatility = float(y.std() / y.mean()) if y.mean() > 0 else 0.0

    # Recent growth: last quarter vs prior quarter
    if n >= 6:
        recent_growth = float((y[-3:].mean() - y[-6:-3].mean()) / (y[-6:-3].mean() + 1e-9))
    else:
        recent_growth = 0.0

    # Outlier rate: IQR method
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    outlier_rate = float(np.mean((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)))

    return {
        "history_length": n,
        "frequency": freq,
        "trend_strength": round(trend_strength, 3),
        "seasonality_strength": round(seasonality_strength, 3),
        "volatility": round(volatility, 3),
        "recent_growth": round(recent_growth, 3),
        "outlier_rate": round(outlier_rate, 3),
        "mean": round(float(y.mean()), 2),
        "std": round(float(y.std()), 2),
    }


def describe_profile(profile: dict) -> str:
    vol = "high" if profile["volatility"] > 0.15 else "low"
    seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"
    trend = "strong" if profile["trend_strength"] > 0.6 else "moderate" if profile["trend_strength"] > 0.3 else "weak"
    return f"{vol} volatility · {seas} seasonality · {trend} trend · {profile['history_length']} months history"
