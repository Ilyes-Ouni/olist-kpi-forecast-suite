from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastArtifacts:
    model: GradientBoostingRegressor
    forecast_actuals: pd.DataFrame
    future_forecast: pd.DataFrame
    metrics: dict[str, float]


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


def train_forecasting_model(daily_sales: pd.DataFrame, train_cutoff_date: str, forecast_horizon_days: int) -> ForecastArtifacts:
    featured = create_time_features(daily_sales)
    featured = featured.dropna().reset_index(drop=True)

    feature_columns = [
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

    train_cutoff = pd.Timestamp(train_cutoff_date)
    train_df = featured[featured["date"] < train_cutoff].copy()
    test_df = featured[featured["date"] >= train_cutoff].copy()

    model = GradientBoostingRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=3,
        min_samples_split=4,
        random_state=42,
    )
    model.fit(train_df[feature_columns], train_df["daily_revenue"])

    test_predictions = model.predict(test_df[feature_columns])
    forecast_actuals = test_df[["date", "daily_revenue"]].copy()
    forecast_actuals["predicted_revenue"] = test_predictions
    forecast_actuals["absolute_error"] = (forecast_actuals["daily_revenue"] - forecast_actuals["predicted_revenue"]).abs()

    mae = mean_absolute_error(test_df["daily_revenue"], test_predictions)
    rmse = float(np.sqrt(mean_squared_error(test_df["daily_revenue"], test_predictions)))
    denominator = np.where(test_df["daily_revenue"] == 0, 1, test_df["daily_revenue"])
    mape = np.mean(np.abs((test_df["daily_revenue"] - test_predictions) / denominator)) * 100

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
                    "date": next_date,
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
        prediction = float(model.predict(row[feature_columns])[0])
        prediction = max(prediction, 0.0)
        future_rows.append({"date": next_date, "predicted_revenue": prediction})
        history.loc[len(history)] = {"date": next_date, "daily_revenue": prediction}

    future_forecast = pd.DataFrame(future_rows)
    metrics = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
    return ForecastArtifacts(model=model, forecast_actuals=forecast_actuals, future_forecast=future_forecast, metrics=metrics)


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
