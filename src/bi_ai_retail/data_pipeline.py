from __future__ import annotations

from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import FIGURES_DIR, FORECAST_HORIZON_DAYS, MODELS_DIR, PROCESSED_DIR, RAW_DATA_PATH, RAW_DATA_URL, RAW_DIR, REPORTS_DIR, TRAIN_CUTOFF_DATE
from .modeling import detect_anomalies, segment_products, train_forecasting_model
from .recommendations import build_recommendations
from .reporting import build_markdown_report, save_json


BRAZIL_REGION_MAP = {
    "AC": "North",
    "AL": "Northeast",
    "AM": "North",
    "AP": "North",
    "BA": "Northeast",
    "CE": "Northeast",
    "DF": "Central-West",
    "ES": "Southeast",
    "GO": "Central-West",
    "MA": "Northeast",
    "MG": "Southeast",
    "MS": "Central-West",
    "MT": "Central-West",
    "PA": "North",
    "PB": "Northeast",
    "PE": "Northeast",
    "PI": "Northeast",
    "PR": "South",
    "RJ": "Southeast",
    "RN": "Northeast",
    "RO": "North",
    "RR": "North",
    "RS": "South",
    "SC": "South",
    "SE": "Northeast",
    "SP": "Southeast",
    "TO": "North",
}


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_dataset(force: bool = False) -> None:
    ensure_directories()
    if force or not RAW_DATA_PATH.exists():
        urlretrieve(RAW_DATA_URL, RAW_DATA_PATH)


def classify_customer_segment(recency_score: int, frequency_score: int, monetary_score: int) -> str:
    if recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4:
        return "Champions"
    if recency_score >= 3 and frequency_score >= 3:
        return "Loyal"
    if recency_score >= 4 and frequency_score <= 2:
        return "Promising"
    if recency_score <= 2 and frequency_score >= 3:
        return "Needs attention"
    return "At risk"


def load_raw_olist() -> pd.DataFrame:
    download_dataset()
    return pd.read_csv(RAW_DATA_PATH, low_memory=False)


def impute_and_engineer(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()

    datetime_columns = [
        "shipping_limit_date",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    missing_before = df.isna().sum().reset_index()
    missing_before.columns = ["feature", "missing_values_before"]

    duplicate_rows_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    text_columns = ["review_comment_title", "review_comment_message", "product_category_name_english", "payment_type", "customer_city", "seller_city"]
    for column in text_columns:
        df[column] = df[column].fillna("Unknown")

    numeric_columns = [
        "review_score",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "payment_installments",
        "payment_value",
        "delivery_days",
        "order_freight_ratio",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    financial_columns = ["price", "freight_value", "payment_value", "total_order_value"]
    for column in financial_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    df = df[df["price"] > 0].copy()
    df = df[df["payment_value"] > 0].copy()
    df = df[df["freight_value"] >= 0].copy()

    noise_before = len(df)
    numeric_noise_cols = ["price", "freight_value", "payment_value", "delivery_days", "product_weight_g"]
    noise_mask = pd.Series(False, index=df.index)
    for column in numeric_noise_cols:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        noise_mask = noise_mask | (df[column] < lower) | (df[column] > upper)
    df = df[~noise_mask].copy()
    noisy_rows_removed = int(noise_before - len(df))

    df["order_date"] = df["order_purchase_timestamp"].dt.floor("D")
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["year"] = df["order_purchase_timestamp"].dt.year
    df["month_name"] = df["order_purchase_timestamp"].dt.month_name()
    df["quarter"] = "Q" + df["order_purchase_timestamp"].dt.quarter.astype(str)
    df["day_of_week"] = df["order_purchase_timestamp"].dt.day_name()
    df["state"] = df["customer_state"].fillna("Unknown")
    df["region"] = df["state"].map(BRAZIL_REGION_MAP).fillna("Unknown")
    df["category"] = df["product_category_name_english"].fillna("Unknown")
    df["review_score"] = df["review_score"].clip(1, 5)
    df["review_text_length"] = df["review_comment_message"].astype(str).str.len()
    df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    df["product_volume_cm3"] = df["product_volume_cm3"].replace(0, np.nan).fillna(df["product_volume_cm3"].median())
    df["product_density_g_cm3"] = (df["product_weight_g"] / df["product_volume_cm3"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["freight_per_weight"] = (df["freight_value"] / df["product_weight_g"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["payment_per_installment"] = (df["payment_value"] / df["payment_installments"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(df["payment_value"])
    df["approval_lag_hours"] = ((df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600).fillna(0)
    df["delivery_delay_days"] = ((df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400).fillna(0)
    df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)
    df["price_to_freight_ratio"] = (df["price"] / df["freight_value"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["order_value_per_item"] = (df["payment_value"] / df["order_item_id"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(df["payment_value"])

    scaled_features = [
        "price",
        "freight_value",
        "payment_value",
        "delivery_days",
        "product_weight_g",
        "product_volume_cm3",
        "review_score",
        "order_freight_ratio",
        "review_text_length",
    ]
    scaler_input = df[scaled_features].fillna(0)
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    minmax_values = minmax_scaler.fit_transform(scaler_input)
    standard_values = standard_scaler.fit_transform(scaler_input)

    scaler_summary = pd.DataFrame(
        {
            "feature": scaled_features,
            "original_mean": scaler_input.mean().values,
            "original_std": scaler_input.std(ddof=0).values,
            "minmax_min": minmax_values.min(axis=0),
            "minmax_max": minmax_values.max(axis=0),
            "standard_mean": standard_values.mean(axis=0),
            "standard_std": standard_values.std(axis=0),
        }
    )

    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(standard_values)
    pca_projection = pd.DataFrame(
        {
            "order_id": df["order_id"].astype(str).values,
            "state": df["state"].values,
            "region": df["region"].values,
            "category": df["category"].values,
            "payment_type": df["payment_type"].values,
            "price": df["price"].values,
            "payment_value": df["payment_value"].values,
            "review_score": df["review_score"].values,
            "pca_1": pca_components[:, 0],
            "pca_2": pca_components[:, 1],
        }
    )

    minmax_scaled_frame = df[["order_id", "state", "region", "category"]].copy()
    for index, feature in enumerate(scaled_features):
        minmax_scaled_frame[f"{feature}_minmax"] = minmax_values[:, index]
    standard_scaled_frame = df[["order_id", "state", "region", "category"]].copy()
    for index, feature in enumerate(scaled_features):
        standard_scaled_frame[f"{feature}_standard"] = standard_values[:, index]

    missing_after = df.isna().sum().reset_index()
    missing_after.columns = ["feature", "missing_values_after"]
    missing_report = missing_before.merge(missing_after, on="feature", how="outer").fillna(0)
    missing_report["resolved_missing_values"] = missing_report["missing_values_before"] - missing_report["missing_values_after"]

    preprocessing_summary = {
        "dataset_name": "Olist merged e-commerce dataset",
        "rows_before_cleaning": int(len(raw_df)),
        "rows_after_deduplication_and_cleaning": int(len(df)),
        "duplicate_rows_removed": duplicate_rows_removed,
        "noisy_rows_removed": noisy_rows_removed,
        "features_for_scaling": scaled_features,
        "feature_fusion_examples": [
            "product_volume_cm3",
            "product_density_g_cm3",
            "freight_per_weight",
            "payment_per_installment",
            "review_text_length",
            "approval_lag_hours",
            "delivery_delay_days",
            "price_to_freight_ratio",
        ],
        "dimension_reduction": {
            "method": "PCA",
            "components": 2,
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        },
        "scalers_used": ["MinMaxScaler", "StandardScaler"],
    }

    return df, preprocessing_summary, scaler_summary, pca_projection, minmax_scaled_frame, standard_scaled_frame, missing_report


def build_order_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    order_summary = (
        clean_df.groupby("order_id")
        .agg(
            order_date=("order_date", "min"),
            customer_unique_id=("customer_unique_id", "first"),
            state=("state", "first"),
            region=("region", "first"),
            city=("customer_city", "first"),
            order_status=("order_status", "first"),
            items=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            sellers=("seller_id", "nunique"),
            total_revenue=("payment_value", "sum"),
            product_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            avg_review_score=("review_score", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            is_late=("is_late", "max"),
        )
        .reset_index()
    )
    order_summary["estimated_cost"] = order_summary["product_value"] * 0.72
    order_summary["profit"] = order_summary["product_value"] - order_summary["estimated_cost"]
    order_summary["profit_margin_pct"] = np.where(order_summary["product_value"] == 0, 0, order_summary["profit"] / order_summary["product_value"] * 100)
    order_summary["avg_item_price"] = order_summary["product_value"] / order_summary["items"]
    return order_summary


def build_daily_sales(order_summary: pd.DataFrame) -> pd.DataFrame:
    daily_sales = (
        order_summary.groupby("order_date")
        .agg(
            daily_revenue=("total_revenue", "sum"),
            daily_profit=("profit", "sum"),
            daily_quantity=("items", "sum"),
            order_count=("order_id", "nunique"),
            active_customers=("customer_unique_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    daily_sales["avg_order_value"] = daily_sales["daily_revenue"] / daily_sales["order_count"]
    daily_sales["profit_margin_pct"] = np.where(daily_sales["daily_revenue"] == 0, 0, daily_sales["daily_profit"] / daily_sales["daily_revenue"] * 100)
    return daily_sales


def build_customer_summary(order_summary: pd.DataFrame, max_order_date: pd.Timestamp) -> pd.DataFrame:
    customer_summary = (
        order_summary.groupby("customer_unique_id")
        .agg(
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            state=("state", "first"),
            region=("region", "first"),
            city=("city", "first"),
            order_count=("order_id", "nunique"),
            total_revenue=("total_revenue", "sum"),
            total_profit=("profit", "sum"),
            avg_review_score=("avg_review_score", "mean"),
        )
        .reset_index()
    )
    customer_summary["recency_days"] = (max_order_date - customer_summary["last_order_date"]).dt.days
    customer_summary["avg_order_value"] = customer_summary["total_revenue"] / customer_summary["order_count"]
    customer_summary["profit_margin_pct"] = np.where(customer_summary["total_revenue"] == 0, 0, customer_summary["total_profit"] / customer_summary["total_revenue"] * 100)
    customer_summary["recency_score"] = pd.qcut(customer_summary["recency_days"].rank(method="first", ascending=True), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    customer_summary["frequency_score"] = pd.qcut(customer_summary["order_count"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_summary["monetary_score"] = pd.qcut(customer_summary["total_revenue"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_summary["rfm_segment"] = customer_summary.apply(
        lambda row: classify_customer_segment(row["recency_score"], row["frequency_score"], row["monetary_score"]),
        axis=1,
    )
    return customer_summary


def build_product_segments(clean_df: pd.DataFrame) -> pd.DataFrame:
    monthly_revenue = clean_df.groupby(["product_id", "category", "order_month"])["payment_value"].sum().reset_index()
    variability = monthly_revenue.groupby(["product_id", "category"])["payment_value"].std().fillna(0).reset_index(name="revenue_volatility")
    product_frame = (
        clean_df.groupby(["product_id", "category"])
        .agg(
            total_revenue=("payment_value", "sum"),
            total_quantity=("order_item_id", "count"),
            avg_unit_price=("price", "mean"),
            transaction_count=("order_id", "nunique"),
            active_months=("order_month", "nunique"),
        )
        .reset_index()
        .rename(columns={"product_id": "product_id", "category": "Category"})
    )
    product_frame["avg_monthly_revenue"] = product_frame["total_revenue"] / product_frame["active_months"]
    product_frame = product_frame.merge(variability.rename(columns={"product_id": "product_id", "category": "Category"}), on=["product_id", "Category"], how="left")
    product_frame["revenue_volatility"] = product_frame["revenue_volatility"].fillna(0)
    product_frame["Description"] = product_frame["product_id"]
    segmented = segment_products(product_frame.rename(columns={"product_id": "StockCode"}))
    segmented = segmented.rename(columns={"StockCode": "product_id"})
    if "Description" in segmented.columns:
        segmented = segmented.drop(columns=["Description"])
    return segmented


def build_aggregates(clean_df: pd.DataFrame, order_summary: pd.DataFrame, customer_summary: pd.DataFrame, product_segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    state_perf = (
        order_summary.groupby(["state", "region"])
        .agg(
            revenue=("total_revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    state_perf["profit_margin_pct"] = np.where(state_perf["revenue"] == 0, 0, state_perf["profit"] / state_perf["revenue"] * 100)

    category_perf = (
        clean_df.groupby("category")
        .agg(
            revenue=("payment_value", "sum"),
            avg_price=("price", "mean"),
            avg_review_score=("review_score", "mean"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"category": "Category"})
        .sort_values("revenue", ascending=False)
    )

    monthly_perf = (
        order_summary.assign(Month=order_summary["order_date"].dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(
            revenue=("total_revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            late_orders=("is_late", "sum"),
        )
        .reset_index()
    )
    monthly_perf["month_date"] = pd.to_datetime(monthly_perf["Month"] + "-01")
    monthly_perf["late_order_rate_pct"] = np.where(monthly_perf["orders"] == 0, 0, monthly_perf["late_orders"] / monthly_perf["orders"] * 100)

    customer_segment_summary = (
        customer_summary.groupby("rfm_segment")
        .agg(customers=("customer_unique_id", "nunique"), revenue=("total_revenue", "sum"), avg_order_value=("avg_order_value", "mean"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    product_segment_summary = (
        product_segments.groupby("segment_name")
        .agg(products=("product_id", "nunique"), revenue=("total_revenue", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    top_products = product_segments.sort_values("total_revenue", ascending=False).head(25).copy()

    return {
        "state_performance": state_perf,
        "category_performance": category_perf,
        "monthly_performance": monthly_perf.sort_values("month_date"),
        "customer_segment_summary": customer_segment_summary,
        "product_segment_summary": product_segment_summary,
        "top_products": top_products,
    }


def build_semantic_outputs(clean_df: pd.DataFrame, order_summary: pd.DataFrame, customer_summary: pd.DataFrame, product_segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fact_sales = clean_df[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_date",
            "state",
            "region",
            "category",
            "price",
            "freight_value",
            "payment_value",
            "review_score",
            "delivery_days",
            "is_late",
            "product_volume_cm3",
            "product_density_g_cm3",
        ]
    ].copy()
    fact_orders = order_summary.copy()
    dim_customer = customer_summary.copy()
    dim_product = product_segments.copy()
    dim_date = pd.DataFrame({"date": pd.date_range(clean_df["order_date"].min(), clean_df["order_date"].max(), freq="D")})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["month"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.month_name()
    dim_date["quarter"] = "Q" + dim_date["date"].dt.quarter.astype(str)
    dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    return {
        "fact_sales": fact_sales,
        "fact_orders": fact_orders,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_date": dim_date,
    }


def build_kpis(clean_df: pd.DataFrame, order_summary: pd.DataFrame, customer_summary: pd.DataFrame, daily_sales: pd.DataFrame, forecast_actuals: pd.DataFrame, forecast_future: pd.DataFrame, anomalies: pd.DataFrame, state_perf: pd.DataFrame, category_perf: pd.DataFrame, preprocessing_summary: dict[str, object]) -> dict[str, float | int | str]:
    total_revenue = float(order_summary["total_revenue"].sum())
    total_profit = float(order_summary["profit"].sum())
    orders_count = int(order_summary["order_id"].nunique())
    customers_count = int(customer_summary["customer_unique_id"].nunique())
    states_count = int(clean_df["state"].nunique())
    average_order_value = float(total_revenue / max(orders_count, 1))
    gross_margin_pct = float(total_profit / max(total_revenue, 1) * 100)
    repeat_customer_rate = float(customer_summary["order_count"].gt(1).mean() * 100)
    latest_30 = daily_sales.tail(30)["daily_revenue"].sum()
    previous_30 = daily_sales.tail(60).head(30)["daily_revenue"].sum()
    sales_growth_pct = float((latest_30 - previous_30) / max(previous_30, 1) * 100)
    forecast_30d_revenue = float(forecast_future["predicted_revenue"].sum())
    forecast_delta_pct = float((forecast_30d_revenue - latest_30) / max(latest_30, 1) * 100)
    forecast_accuracy_pct = float(100 - forecast_actuals["absolute_error"].sum() / max(forecast_actuals["daily_revenue"].sum(), 1) * 100)
    late_order_rate_pct = float(order_summary["is_late"].mean() * 100)

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "orders_count": orders_count,
        "customers_count": customers_count,
        "states_count": states_count,
        "average_order_value": average_order_value,
        "gross_margin_pct": gross_margin_pct,
        "repeat_customer_rate": repeat_customer_rate,
        "sales_growth_pct": sales_growth_pct,
        "forecast_30d_revenue": forecast_30d_revenue,
        "forecast_delta_pct": forecast_delta_pct,
        "forecast_accuracy_pct": forecast_accuracy_pct,
        "anomaly_days": int(len(anomalies)),
        "late_order_rate_pct": late_order_rate_pct,
        "top_state": str(state_perf.iloc[0]["state"]),
        "top_category": str(category_perf.iloc[0]["Category"]),
        "rows_after_cleaning": int(preprocessing_summary["rows_after_deduplication_and_cleaning"]),
        "noisy_rows_removed": int(preprocessing_summary["noisy_rows_removed"]),
        "date_range_start": clean_df["order_date"].min().strftime("%Y-%m-%d"),
        "date_range_end": clean_df["order_date"].max().strftime("%Y-%m-%d"),
    }


def run_pipeline() -> dict[str, object]:
    raw_df = load_raw_olist()
    clean_df, preprocessing_summary, scaler_summary, pca_projection, minmax_scaled_frame, standard_scaled_frame, missing_report = impute_and_engineer(raw_df)
    order_summary = build_order_summary(clean_df)
    daily_sales = build_daily_sales(order_summary)
    customer_summary = build_customer_summary(order_summary, order_summary["order_date"].max())
    product_segments = build_product_segments(clean_df)
    forecast_artifacts = train_forecasting_model(daily_sales=daily_sales, train_cutoff_date=TRAIN_CUTOFF_DATE, forecast_horizon_days=FORECAST_HORIZON_DAYS)
    anomalies = detect_anomalies(daily_sales)
    aggregates = build_aggregates(clean_df, order_summary, customer_summary, product_segments)
    kpis = build_kpis(
        clean_df,
        order_summary,
        customer_summary,
        daily_sales,
        forecast_artifacts.forecast_actuals,
        forecast_artifacts.future_forecast,
        anomalies,
        aggregates["state_performance"],
        aggregates["category_performance"],
        preprocessing_summary,
    )
    recommendations = build_recommendations(
        kpis=kpis,
        future_forecast=forecast_artifacts.future_forecast,
        recent_daily_sales=daily_sales,
        anomalies=anomalies,
        product_segments=product_segments,
        category_performance=aggregates["category_performance"],
        customer_segments=aggregates["customer_segment_summary"],
        country_performance=aggregates["state_performance"].rename(columns={"state": "Country"}),
    )
    semantic_outputs = build_semantic_outputs(clean_df, order_summary, customer_summary, product_segments)

    clean_df.to_csv(PROCESSED_DIR / "clean_retail.csv", index=False)
    order_summary.to_csv(PROCESSED_DIR / "order_summary.csv", index=False)
    customer_summary.to_csv(PROCESSED_DIR / "customer_summary.csv", index=False)
    daily_sales.to_csv(PROCESSED_DIR / "daily_sales.csv", index=False)
    forecast_artifacts.forecast_actuals.to_csv(PROCESSED_DIR / "forecast_actuals.csv", index=False)
    forecast_artifacts.future_forecast.to_csv(PROCESSED_DIR / "future_forecast.csv", index=False)
    anomalies.to_csv(PROCESSED_DIR / "anomalies.csv", index=False)
    product_segments.to_csv(PROCESSED_DIR / "product_segments.csv", index=False)
    recommendations.to_csv(PROCESSED_DIR / "recommendations.csv", index=False)
    scaler_summary.to_csv(PROCESSED_DIR / "scaler_summary.csv", index=False)
    pca_projection.to_csv(PROCESSED_DIR / "pca_projection.csv", index=False)
    minmax_scaled_frame.to_csv(PROCESSED_DIR / "minmax_scaled_features.csv", index=False)
    standard_scaled_frame.to_csv(PROCESSED_DIR / "standard_scaled_features.csv", index=False)
    missing_report.to_csv(PROCESSED_DIR / "missing_value_report.csv", index=False)

    for name, frame in aggregates.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    for name, frame in semantic_outputs.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    save_json(PROCESSED_DIR / "kpis.json", kpis)
    save_json(PROCESSED_DIR / "model_metrics.json", forecast_artifacts.metrics)
    save_json(PROCESSED_DIR / "preprocessing_summary.json", preprocessing_summary)
    build_markdown_report(REPORTS_DIR / "project_summary.md", kpis, forecast_artifacts.metrics, recommendations)
    joblib.dump(forecast_artifacts.model, MODELS_DIR / "forecast_model.joblib")

    return {
        "clean_df": clean_df,
        "order_summary": order_summary,
        "customer_summary": customer_summary,
        "daily_sales": daily_sales,
        "forecast_actuals": forecast_artifacts.forecast_actuals,
        "future_forecast": forecast_artifacts.future_forecast,
        "anomalies": anomalies,
        "product_segments": product_segments,
        "aggregates": aggregates,
        "recommendations": recommendations,
        "kpis": kpis,
        "model_metrics": forecast_artifacts.metrics,
        "semantic_outputs": semantic_outputs,
        "preprocessing_summary": preprocessing_summary,
        "scaler_summary": scaler_summary,
        "pca_projection": pca_projection,
        "missing_report": missing_report,
    }
