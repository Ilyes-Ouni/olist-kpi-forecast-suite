"""
Multi-model daily revenue forecasting (academic BI + AI project).

We train three candidates on the same train/test split:
  - GradientBoostingRegressor (sklearn baseline)
  - XGBoost
  - Prophet (trend + seasonality; skipped cleanly if not installed)

Selection rule: lowest RMSE on the hold-out window, MAPE as tie-breaker.
Only the winner is used for the official 30-day future forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local install
    XGBRegressor = None  # type: ignore[misc, assignment]
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local install
    Prophet = None  # type: ignore[misc, assignment]
    PROPHET_AVAILABLE = False


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7_mean",
    "rolling_30_mean",
    "rolling_7_std",
]


@dataclass
class ForecastArtifacts:
    model: Any
    model_name: str
    forecast_actuals: pd.DataFrame
    future_forecast: pd.DataFrame
    metrics: dict[str, Any]
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_actuals: pd.DataFrame = field(default_factory=pd.DataFrame)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["day_of_month"] = featured["date"].dt.day
    featured["week_of_year"] = featured["date"].dt.isocalendar().week.astype(int)
    featured["month"] = featured["date"].dt.month
    featured["quarter"] = featured["date"].dt.quarter
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)
    featured["lag_1"] = featured["daily_revenue"].shift(1)
    featured["lag_7"] = featured["daily_revenue"].shift(7)
    featured["rolling_7_mean"] = featured["daily_revenue"].shift(1).rolling(7).mean()
    featured["rolling_30_mean"] = featured["daily_revenue"].shift(1).rolling(30).mean()
    featured["rolling_7_std"] = featured["daily_revenue"].shift(1).rolling(7).std()
    return featured


def _regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    denominator = np.where(y_true_arr == 0, 1.0, y_true_arr)
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100)
    r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}


def _recursive_tree_forecast(
    model: Any,
    daily_sales: pd.DataFrame,
    forecast_horizon_days: int,
) -> pd.DataFrame:
    """Multi-step ahead forecast for lag-based tree models (GBR / XGBoost)."""
    history = daily_sales[["date", "daily_revenue"]].copy().sort_values("date").reset_index(drop=True)
    future_rows: list[dict[str, float | pd.Timestamp]] = []

    for _ in range(forecast_horizon_days):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        lag_1 = float(history["daily_revenue"].iloc[-1])
        lag_7 = float(history["daily_revenue"].iloc[-7]) if len(history) >= 7 else lag_1
        rolling_7 = float(history["daily_revenue"].tail(7).mean())
        rolling_30 = float(history["daily_revenue"].tail(30).mean())
        rolling_std = float(history["daily_revenue"].tail(7).std(ddof=0))

        row = pd.DataFrame(
            [
                {
                    "day_of_week": next_date.dayofweek,
                    "day_of_month": next_date.day,
                    "week_of_year": int(next_date.isocalendar().week),
                    "month": next_date.month,
                    "quarter": next_date.quarter,
                    "is_weekend": int(next_date.dayofweek in [5, 6]),
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "rolling_7_mean": rolling_7,
                    "rolling_30_mean": rolling_30,
                    "rolling_7_std": rolling_std,
                }
            ]
        )
        prediction = float(model.predict(row[FEATURE_COLUMNS])[0])
        prediction = max(prediction, 0.0)
        future_rows.append({"date": next_date, "predicted_revenue": prediction})
        history.loc[len(history)] = {"date": next_date, "daily_revenue": prediction}

    return pd.DataFrame(future_rows)


def _fit_gbr(train_df: pd.DataFrame) -> Any:
    model = GradientBoostingRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=3,
        min_samples_split=4,
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["daily_revenue"])
    return model


def _fit_xgboost(train_df: pd.DataFrame) -> Any:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["daily_revenue"])
    return model


def _fit_prophet(train_raw: pd.DataFrame) -> Any:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("prophet is not installed")
    # Prophet likes quiet logs for coursework demos.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    prophet_train = train_raw.rename(columns={"date": "ds", "daily_revenue": "y"})[["ds", "y"]]
    model.fit(prophet_train)
    return model


def _prophet_predict(model: Any, dates: pd.Series) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    forecast = model.predict(future)
    return np.maximum(forecast["yhat"].to_numpy(dtype=float), 0.0)


def _select_best_model(comparison: pd.DataFrame) -> str:
    """Primary key = RMSE; tie-break = MAPE."""
    ranked = comparison.sort_values(["rmse", "mape"], ascending=[True, True]).reset_index(drop=True)
    return str(ranked.iloc[0]["model"])


def train_forecasting_model(
    daily_sales: pd.DataFrame,
    train_cutoff_date: str,
    forecast_horizon_days: int,
) -> ForecastArtifacts:
    featured = create_time_features(daily_sales).dropna().reset_index(drop=True)
    train_cutoff = pd.Timestamp(train_cutoff_date)
    train_df = featured[featured["date"] < train_cutoff].copy()
    test_df = featured[featured["date"] >= train_cutoff].copy()

    raw = daily_sales[["date", "daily_revenue"]].copy().sort_values("date").reset_index(drop=True)
    train_raw = raw[raw["date"] < train_cutoff].copy()
    test_raw = raw[raw["date"] >= train_cutoff].copy()

    candidates: dict[str, dict[str, Any]] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    notes: list[str] = []

    # --- Gradient Boosting baseline ---
    gbr = _fit_gbr(train_df)
    gbr_pred = np.maximum(gbr.predict(test_df[FEATURE_COLUMNS]), 0.0)
    candidates["GradientBoosting"] = {
        "model": gbr,
        "metrics": _regression_metrics(test_df["daily_revenue"], gbr_pred),
        "future_fn": lambda m=gbr: _recursive_tree_forecast(m, daily_sales, forecast_horizon_days),
        "family": "tree",
    }
    candidate_predictions["pred_GradientBoosting"] = gbr_pred

    # --- XGBoost ---
    if XGBOOST_AVAILABLE:
        try:
            xgb_model = _fit_xgboost(train_df)
            xgb_pred = np.maximum(xgb_model.predict(test_df[FEATURE_COLUMNS]), 0.0)
            candidates["XGBoost"] = {
                "model": xgb_model,
                "metrics": _regression_metrics(test_df["daily_revenue"], xgb_pred),
                "future_fn": lambda m=xgb_model: _recursive_tree_forecast(m, daily_sales, forecast_horizon_days),
                "family": "tree",
            }
            candidate_predictions["pred_XGBoost"] = xgb_pred
        except Exception as exc:  # pragma: no cover
            notes.append(f"XGBoost skipped: {exc}")
    else:
        notes.append("XGBoost skipped: package not installed")

    # --- Prophet ---
    if PROPHET_AVAILABLE:
        try:
            prophet_model = _fit_prophet(train_raw)
            # Evaluate on the same calendar dates as the featured test window.
            prophet_pred = _prophet_predict(prophet_model, test_df["date"])
            candidates["Prophet"] = {
                "model": prophet_model,
                "metrics": _regression_metrics(test_df["daily_revenue"].to_numpy(), prophet_pred),
                "future_fn": lambda m=prophet_model: _prophet_future(m, daily_sales, forecast_horizon_days),
                "family": "prophet",
            }
            candidate_predictions["pred_Prophet"] = prophet_pred
        except Exception as exc:  # pragma: no cover
            notes.append(f"Prophet skipped: {exc}")
    else:
        notes.append("Prophet skipped: package not installed")

    if not candidates:
        raise RuntimeError("No forecasting models could be trained.")

    comparison_rows = []
    for name, payload in candidates.items():
        row = {"model": name, **payload["metrics"]}
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison["selected"] = False

    winner_name = _select_best_model(comparison)
    comparison.loc[comparison["model"] == winner_name, "selected"] = True
    comparison = comparison.sort_values(["selected", "rmse"], ascending=[False, True]).reset_index(drop=True)

    winner = candidates[winner_name]
    winner_pred = candidate_predictions[f"pred_{winner_name}"]

    forecast_actuals = test_df[["date", "daily_revenue"]].copy()
    forecast_actuals["predicted_revenue"] = winner_pred
    forecast_actuals["absolute_error"] = (forecast_actuals["daily_revenue"] - forecast_actuals["predicted_revenue"]).abs()
    forecast_actuals["model"] = winner_name

    candidate_actuals = test_df[["date", "daily_revenue"]].copy()
    for col_name, preds in candidate_predictions.items():
        candidate_actuals[col_name] = preds

    future_forecast = winner["future_fn"]()
    future_forecast["model"] = winner_name

    metrics = {
        **winner["metrics"],
        "selected_model": winner_name,
        "models_compared": float(len(candidates)),
        "selection_rule": "lowest_rmse_then_mape",
    }
    # Keep numeric metric keys used elsewhere as floats from the winner.
    for key in ("mae", "mse", "rmse", "mape", "r2"):
        metrics[key] = float(winner["metrics"][key])

    if notes:
        metrics["model_notes"] = " | ".join(notes)

    return ForecastArtifacts(
        model=winner["model"],
        model_name=winner_name,
        forecast_actuals=forecast_actuals,
        future_forecast=future_forecast,
        metrics=metrics,
        comparison=comparison,
        candidate_actuals=candidate_actuals,
    )


def _prophet_future(model: Any, daily_sales: pd.DataFrame, forecast_horizon_days: int) -> pd.DataFrame:
    last_date = pd.to_datetime(daily_sales["date"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
    preds = _prophet_predict(model, pd.Series(future_dates))
    return pd.DataFrame({"date": future_dates, "predicted_revenue": preds})


def detect_anomalies(daily_sales: pd.DataFrame) -> pd.DataFrame:
    anomalies = daily_sales.copy().sort_values("date").reset_index(drop=True)
    anomalies["rolling_7_mean"] = anomalies["daily_revenue"].rolling(7, min_periods=3).mean()
    anomalies["rolling_7_std"] = anomalies["daily_revenue"].rolling(7, min_periods=3).std().fillna(0)
    anomalies["revenue_pct_change"] = anomalies["daily_revenue"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    feature_frame = anomalies[["daily_revenue", "daily_quantity", "avg_order_value", "rolling_7_mean", "rolling_7_std", "revenue_pct_change"]].fillna(0)
    model = IsolationForest(contamination=0.03, random_state=42)
    anomalies["anomaly_score"] = model.fit_predict(feature_frame)

    overall_mean = anomalies["daily_revenue"].mean()
    overall_std = anomalies["daily_revenue"].std(ddof=0) or 1
    anomalies["z_score"] = (anomalies["daily_revenue"] - overall_mean) / overall_std
    anomalies["anomaly_flag"] = ((anomalies["anomaly_score"] == -1) | (anomalies["z_score"].abs() >= 2.5)).astype(int)
    anomalies["anomaly_type"] = np.where(
        anomalies["daily_revenue"] >= anomalies["rolling_7_mean"],
        "Spike",
        "Drop",
    )
    return anomalies[anomalies["anomaly_flag"] == 1].copy()


def segment_products(product_frame: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    segment_base = product_frame.copy()
    feature_columns = [
        "total_revenue",
        "total_quantity",
        "avg_unit_price",
        "transaction_count",
        "active_months",
        "avg_monthly_revenue",
        "revenue_volatility",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(segment_base[feature_columns].fillna(0))

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    segment_base["cluster_id"] = model.fit_predict(scaled)

    cluster_profile = segment_base.groupby("cluster_id")[["total_revenue", "revenue_volatility"]].mean().reset_index()
    top_revenue_cluster = int(cluster_profile.sort_values("total_revenue", ascending=False).iloc[0]["cluster_id"])
    top_volatility_cluster = int(cluster_profile.sort_values("revenue_volatility", ascending=False).iloc[0]["cluster_id"])

    labels: dict[int, str] = {}
    for _, row in cluster_profile.iterrows():
        cluster_id = int(row["cluster_id"])
        if cluster_id == top_revenue_cluster:
            labels[cluster_id] = "High performers"
        elif cluster_id == top_volatility_cluster:
            labels[cluster_id] = "Seasonal opportunities"
        else:
            labels[cluster_id] = "At-risk products"

    segment_base["segment_name"] = segment_base["cluster_id"].map(labels)
    return segment_base
