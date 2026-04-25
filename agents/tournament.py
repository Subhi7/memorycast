import numpy as np
import pandas as pd
import lightgbm as lgb
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from mlforecast import MLForecast


SEASON_LENGTH = 12
N_WINDOWS = 3
_LGBM_PARAMS = dict(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=42, verbose=-1)


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


def _run_mlforecast(df: pd.DataFrame, horizon: int) -> dict:
    mlf = MLForecast(
        models=[lgb.LGBMRegressor(**_LGBM_PARAMS)],
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        date_features=["month"],
    )
    cv = mlf.cross_validation(df=df, h=horizon, n_windows=N_WINDOWS, step_size=horizon)
    col = "LGBMRegressor"
    return {
        "LightGBM": {
            "wape": round(_wape(cv["y"].values, cv[col].values), 4),
            "bias": round(_bias(cv["y"].values, cv[col].values), 4),
            "type": "ml",
        }
    }


def run_tournament(df: pd.DataFrame, horizon: int = 3) -> tuple[dict, str]:
    """Run all models, return (results_dict, winner_name)."""
    results = {}
    results.update(_run_statsforecast(df, horizon))
    results.update(_run_mlforecast(df, horizon))
    winner = min(results, key=lambda k: (results[k]["wape"], abs(results[k]["bias"])))
    return results, winner


def run_final_forecast(df: pd.DataFrame, horizon: int, winner: str) -> pd.DataFrame:
    """Fit winner on full history, return forecast DataFrame with [ds, forecast]."""
    uid = df["unique_id"].iloc[0]

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

    # LightGBM
    mlf = MLForecast(
        models=[lgb.LGBMRegressor(**_LGBM_PARAMS)],
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        date_features=["month"],
    )
    mlf.fit(df)
    fc = mlf.predict(h=horizon).reset_index()
    return fc[["ds", "LGBMRegressor"]].rename(columns={"LGBMRegressor": "forecast"})
