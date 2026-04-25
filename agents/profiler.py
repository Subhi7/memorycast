import numpy as np
import pandas as pd


def profile_series(df: pd.DataFrame, freq: str = "MS") -> dict:
    """Return a rich profile dict for a time series DataFrame with columns [ds, y]."""
    y = df["y"].values.astype(float)
    n = len(y)
    season_lag = 12

    # ── Trend ─────────────────────────────────────────────────────────────────
    x = np.arange(n)
    coeffs = np.polyfit(x, y, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    trend_strength = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    trend_slope = float(coeffs[0])                      # units/month
    trend_direction = "up" if trend_slope > 0.001 * y.mean() else "down" if trend_slope < -0.001 * y.mean() else "flat"

    # ── Seasonality ───────────────────────────────────────────────────────────
    if n > season_lag * 2:
        seasonality_strength = float(max(0.0, pd.Series(y).autocorr(lag=season_lag)))
    else:
        seasonality_strength = 0.0

    # Peak month (1-12): which calendar month averages highest
    if "ds" in df.columns and n >= season_lag:
        monthly_means = df.groupby(df["ds"].dt.month)["y"].mean()
        peak_month = int(monthly_means.idxmax())
        seasonal_amplitude = float(
            (monthly_means.max() - monthly_means.min()) / (y.mean() + 1e-9)
        )
    else:
        peak_month = 0
        seasonal_amplitude = 0.0

    # ── Volatility ────────────────────────────────────────────────────────────
    volatility = float(y.std() / y.mean()) if y.mean() > 0 else 0.0

    # ── Autocorrelation at lag 1 (AR signal) ──────────────────────────────────
    acf_lag1 = float(pd.Series(y).autocorr(lag=1)) if n > 10 else 0.0

    # ── Recent momentum ───────────────────────────────────────────────────────
    if n >= 6:
        recent_growth = float((y[-3:].mean() - y[-6:-3].mean()) / (y[-6:-3].mean() + 1e-9))
    else:
        recent_growth = 0.0

    # Year-over-year change (last 12m vs prior 12m)
    if n >= 24:
        yoy_change = float((y[-12:].mean() - y[-24:-12].mean()) / (y[-24:-12].mean() + 1e-9))
    else:
        yoy_change = recent_growth

    # ── Distribution ─────────────────────────────────────────────────────────
    # Outlier rate: IQR method
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    outlier_rate = float(np.mean((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)))

    # Skewness: positive = right tail (spikes up), negative = left tail
    if y.std() > 0:
        skewness = float(np.mean(((y - y.mean()) / y.std()) ** 3))
    else:
        skewness = 0.0

    # ── Stationarity proxy (ADF-lite) ─────────────────────────────────────────
    # Simple variance-ratio test: ratio of first-half vs second-half variance
    # Values near 1 = stationary, far from 1 = non-stationary
    half = n // 2
    var_ratio = float(y[:half].var() / (y[half:].var() + 1e-9)) if half > 2 else 1.0
    is_stationary = 0.5 < var_ratio < 2.0

    # ── Regime change detection ───────────────────────────────────────────────
    # CUSUM: did the mean shift significantly in the last third?
    third = n // 3
    if third > 3:
        early_mean = y[:third].mean()
        late_mean = y[-third:].mean()
        regime_shift = float(abs(late_mean - early_mean) / (y.std() + 1e-9))
    else:
        regime_shift = 0.0

    return {
        # Core profile (used for memory similarity matching)
        "history_length": n,
        "frequency": freq,
        "volatility": round(volatility, 3),
        "seasonality_strength": round(seasonality_strength, 3),
        "trend_strength": round(trend_strength, 3),
        # Extended profile (used by Claude agent for richer reasoning)
        "trend_direction": trend_direction,
        "trend_slope_pct": round(trend_slope / (y.mean() + 1e-9) * 100, 3),  # % change per month
        "peak_month": peak_month,
        "seasonal_amplitude": round(seasonal_amplitude, 3),
        "acf_lag1": round(acf_lag1, 3),
        "recent_growth": round(recent_growth, 3),
        "yoy_change": round(yoy_change, 3),
        "outlier_rate": round(outlier_rate, 3),
        "skewness": round(skewness, 3),
        "is_stationary": is_stationary,
        "regime_shift": round(regime_shift, 3),
        "mean": round(float(y.mean()), 2),
        "std": round(float(y.std()), 2),
    }


def describe_profile(profile: dict) -> str:
    parts = []

    vol = "high" if profile["volatility"] > 0.15 else "low"
    parts.append(f"{vol} volatility")

    seas = "strong" if profile["seasonality_strength"] > 0.4 else "weak"
    parts.append(f"{seas} seasonality")

    trend = "strong" if profile["trend_strength"] > 0.6 else "moderate" if profile["trend_strength"] > 0.3 else "weak"
    direction = profile.get("trend_direction", "")
    parts.append(f"{trend}-{direction} trend" if direction else f"{trend} trend")

    if profile.get("regime_shift", 0) > 1.5:
        parts.append("regime shift detected")

    if profile.get("outlier_rate", 0) > 0.1:
        parts.append(f"{profile['outlier_rate']:.0%} outliers")

    if profile.get("skewness", 0) > 1.5:
        parts.append("right-skewed")
    elif profile.get("skewness", 0) < -1.5:
        parts.append("left-skewed")

    parts.append(f"{profile['history_length']} months history")
    return " · ".join(parts)
