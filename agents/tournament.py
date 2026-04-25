import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

SEASON_LENGTH = 12
N_WINDOWS = 3
_ML_NAME = "GradientBoosting"

# To swap in LightGBM once libomp is installed (brew install libomp):
# import lightgbm as lgb
# _ML_MODEL = lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=42, verbose=-1)
# _ML_NAME = "LightGBM"


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("inf")


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred.sum() - y_true.sum()) / (y_true.sum() + 1e-9))


def _run_statsforecast(df: pd.DataFrame, horizon: int) -> dict:
    models = [
        SeasonalNaive(season_length=SEASON_LENGTH),
        AutoARIMA(season_length=SEASON_LENGTH, max_p=3, max_q=3, approximation=True),
        AutoETS(season_length=SEASON_LENGTH),
    ]
    sf = StatsForecast(models=models, freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=horizon, n_windows=N_WINDOWS, step_size=horizon)

    results = {}
    for col in ["SeasonalNaive", "AutoARIMA", "AutoETS"]:
        results[col] = {
            "wape": round(_wape(cv["y"].values, cv[col].values), 4),
            "bias": round(_bias(cv["y"].values, cv[col].values), 4),
            "type": "statistical",
        }
    return results


def _make_lag_features(y: np.ndarray, ds: pd.Series, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y_target) arrays using lag features + month."""
    n = len(y)
    max_lag = max(lags)
    rows_X, rows_y = [], []
    for i in range(max_lag, n):
        feats = [y[i - lag] for lag in lags]
        feats.append(ds.iloc[i].month)
        rows_X.append(feats)
        rows_y.append(y[i])
    return np.array(rows_X), np.array(rows_y)


def _run_ml(df: pd.DataFrame, horizon: int) -> dict:
    """Rolling-window CV for the gradient boosting model with lag features."""
    lags = [1, 2, 3, 6, 12]
    y = df["y"].values.astype(float)
    ds = df["ds"]
    n = len(y)

    preds_all, actuals_all = [], []
    window_size = n - N_WINDOWS * horizon

    for w in range(N_WINDOWS):
        train_end = window_size + w * horizon
        test_start = train_end
        test_end = test_start + horizon

        y_train = y[:train_end]
        ds_train = ds.iloc[:train_end]
        y_test = y[test_start:test_end]

        X_train, y_lag_train = _make_lag_features(y_train, ds_train, lags)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_lag_train)

        # Recursive multi-step forecast
        y_hist = list(y_train)
        ds_future = ds.iloc[test_start:test_end]
        for step in range(horizon):
            feats = [y_hist[-(lag)] for lag in lags] + [ds_future.iloc[step].month]
            pred = float(model.predict([feats])[0])
            preds_all.append(pred)
            y_hist.append(pred)
        actuals_all.extend(y_test.tolist())

    preds = np.array(preds_all)
    actuals = np.array(actuals_all)
    return {
        _ML_NAME: {
            "wape": round(_wape(actuals, preds), 4),
            "bias": round(_bias(actuals, preds), 4),
            "type": "ml",
        }
    }


def run_tournament(df: pd.DataFrame, horizon: int = 3) -> tuple[dict, str]:
    """Run all models, return (results_dict, winner_name)."""
    results = {}
    results.update(_run_statsforecast(df, horizon))
    results.update(_run_ml(df, horizon))
    winner = min(results, key=lambda k: (results[k]["wape"], abs(results[k]["bias"])))
    return results, winner


def run_final_forecast(df: pd.DataFrame, horizon: int, winner: str) -> pd.DataFrame:
    """Fit winner on full history and return forecast DataFrame with [ds, forecast]."""
    if winner in ("SeasonalNaive", "AutoARIMA", "AutoETS"):
        model_map = {
            "SeasonalNaive": SeasonalNaive(season_length=SEASON_LENGTH),
            "AutoARIMA": AutoARIMA(season_length=SEASON_LENGTH, max_p=3, max_q=3, approximation=True),
            "AutoETS": AutoETS(season_length=SEASON_LENGTH),
        }
        sf = StatsForecast(models=[model_map[winner]], freq="MS", n_jobs=1)
        sf.fit(df)
        fc = sf.predict(h=horizon).reset_index()
        return fc[["ds", winner]].rename(columns={winner: "forecast"})

    # GradientBoosting: recursive forecast
    lags = [1, 2, 3, 6, 12]
    y = df["y"].values.astype(float)
    X_train, y_train = _make_lag_features(y, df["ds"], lags)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    future_ds = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq="MS")[1:]
    y_hist = list(y)
    forecasts = []
    for step in range(horizon):
        feats = [y_hist[-(lag)] for lag in lags] + [future_ds[step].month]
        pred = float(model.predict([feats])[0])
        forecasts.append(pred)
        y_hist.append(pred)

    return pd.DataFrame({"ds": future_ds, "forecast": forecasts})
