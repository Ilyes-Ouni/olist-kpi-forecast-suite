from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bi_ai_retail.data_pipeline import run_pipeline


st.set_page_config(page_title="Olist Commerce Decision Lab", layout="wide")


def format_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def load_or_build() -> dict[str, pd.DataFrame | dict]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    required_files = [
        "clean_retail.csv",
        "order_summary.csv",
        "customer_summary.csv",
        "daily_sales.csv",
        "future_forecast.csv",
        "forecast_actuals.csv",
        "anomalies.csv",
        "product_segments.csv",
        "state_performance.csv",
        "category_performance.csv",
        "monthly_performance.csv",
        "customer_segment_summary.csv",
        "product_segment_summary.csv",
        "recommendations.csv",
        "scaler_summary.csv",
        "pca_projection.csv",
        "missing_value_report.csv",
        "preprocessing_summary.json",
        "kpis.json",
        "model_metrics.json",
    ]
    if not all((processed_dir / name).exists() for name in required_files):
        run_pipeline()

    return {
        "clean_retail": pd.read_csv(processed_dir / "clean_retail.csv", parse_dates=["order_purchase_timestamp", "order_date"]),
        "order_summary": pd.read_csv(processed_dir / "order_summary.csv", parse_dates=["order_date"]),
        "customer_summary": pd.read_csv(processed_dir / "customer_summary.csv", parse_dates=["first_order_date", "last_order_date"]),
        "daily_sales": pd.read_csv(processed_dir / "daily_sales.csv", parse_dates=["date"]),
        "future_forecast": pd.read_csv(processed_dir / "future_forecast.csv", parse_dates=["date"]),
        "forecast_actuals": pd.read_csv(processed_dir / "forecast_actuals.csv", parse_dates=["date"]),
        "anomalies": pd.read_csv(processed_dir / "anomalies.csv", parse_dates=["date"]),
        "product_segments": pd.read_csv(processed_dir / "product_segments.csv"),
        "state_performance": pd.read_csv(processed_dir / "state_performance.csv"),
        "category_performance": pd.read_csv(processed_dir / "category_performance.csv"),
        "monthly_performance": pd.read_csv(processed_dir / "monthly_performance.csv", parse_dates=["month_date"]),
        "customer_segment_summary": pd.read_csv(processed_dir / "customer_segment_summary.csv"),
        "product_segment_summary": pd.read_csv(processed_dir / "product_segment_summary.csv"),
        "recommendations": pd.read_csv(processed_dir / "recommendations.csv"),
        "scaler_summary": pd.read_csv(processed_dir / "scaler_summary.csv"),
        "pca_projection": pd.read_csv(processed_dir / "pca_projection.csv"),
        "missing_value_report": pd.read_csv(processed_dir / "missing_value_report.csv"),
        "preprocessing_summary": json.loads((processed_dir / "preprocessing_summary.json").read_text(encoding="utf-8")),
        "kpis": json.loads((processed_dir / "kpis.json").read_text(encoding="utf-8")),
        "model_metrics": json.loads((processed_dir / "model_metrics.json").read_text(encoding="utf-8")),
    }


def build_filtered_views(clean_retail: pd.DataFrame, region_filter: list[str], category_filter: list[str], state_filter: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = clean_retail.copy()
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if state_filter:
        filtered = filtered[filtered["state"].isin(state_filter)]

    orders = (
        filtered.groupby("order_id")
        .agg(
            order_date=("order_date", "min"),
            customer_unique_id=("customer_unique_id", "first"),
            state=("state", "first"),
            region=("region", "first"),
            revenue=("payment_value", "sum"),
            product_value=("price", "sum"),
            avg_review_score=("review_score", "mean"),
        )
        .reset_index()
    )
    return filtered, orders


data = load_or_build()
clean_retail = data["clean_retail"]
kpis = data["kpis"]
metrics = data["model_metrics"]
prep = data["preprocessing_summary"]

if "product_id" not in data["product_segments"].columns and "StockCode" in data["product_segments"].columns:
    data["product_segments"] = data["product_segments"].rename(columns={"StockCode": "product_id"})
if "Description" in data["product_segments"].columns:
    data["product_segments"] = data["product_segments"].drop(columns=["Description"])

st.title("Olist Commerce Decision Lab")
st.caption("BI + AI academic project using the Olist e-commerce dataset with extended preprocessing and feature analysis")

with st.sidebar:
    st.header("Filters")
    region_filter = st.multiselect("Region", sorted(clean_retail["region"].dropna().unique()))
    category_filter = st.multiselect("Category", sorted(clean_retail["category"].dropna().unique()))
    state_filter = st.multiselect("State", sorted(clean_retail["state"].dropna().unique()))

filtered_sales, filtered_orders = build_filtered_views(clean_retail, region_filter, category_filter, state_filter)

metric_cols = st.columns(6)
metric_cols[0].metric("Revenue", format_money(filtered_sales["payment_value"].sum()))
metric_cols[1].metric("Orders", f"{filtered_orders['order_id'].nunique():,}")
metric_cols[2].metric("Customers", f"{filtered_sales['customer_unique_id'].nunique():,}")
metric_cols[3].metric("Avg Review", f"{filtered_sales['review_score'].mean():.2f}")
metric_cols[4].metric("Late Orders", f"{filtered_sales['is_late'].mean() * 100:.1f}%")
metric_cols[5].metric("Next 30 Days", format_money(kpis["forecast_30d_revenue"]))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive BI", "Data Preparation", "Forecasting", "Segmentation", "Recommendations"])

with tab1:
    top_left, top_right = st.columns([2, 1])
    trend = (
        filtered_sales.groupby("order_date")
        .agg(revenue=("payment_value", "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=trend["order_date"], y=trend["revenue"], name="Revenue"))
    trend_fig.add_trace(go.Scatter(x=trend["order_date"], y=trend["orders"], name="Orders", yaxis="y2"))
    trend_fig.update_layout(
        title="Revenue and Order Trend",
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
    )
    top_left.plotly_chart(trend_fig, width="stretch")

    state_summary = (
        filtered_sales.groupby(["state", "region"])["payment_value"]
        .sum()
        .reset_index()
        .sort_values("payment_value", ascending=False)
        .head(10)
    )
    state_fig = px.bar(state_summary, x="state", y="payment_value", color="region", title="Top States by Revenue")
    top_right.plotly_chart(state_fig, width="stretch")

    bottom_left, bottom_right = st.columns(2)
    category_summary = (
        filtered_sales.groupby("category")
        .agg(revenue=("payment_value", "sum"), avg_review=("review_score", "mean"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(15)
    )
    category_fig = px.bar(category_summary, x="category", y="revenue", color="avg_review", title="Category Revenue and Review Quality")
    bottom_left.plotly_chart(category_fig, width="stretch")

    state_scatter = (
        filtered_sales.groupby("state")
        .agg(revenue=("payment_value", "sum"), avg_delivery_days=("delivery_days", "mean"), late_order_rate=("is_late", "mean"))
        .reset_index()
    )
    state_scatter_fig = px.scatter(
        state_scatter,
        x="avg_delivery_days",
        y="revenue",
        size="late_order_rate",
        color="late_order_rate",
        hover_name="state",
        title="State Performance vs Delivery Speed",
    )
    bottom_right.plotly_chart(state_scatter_fig, width="stretch")

with tab2:
    st.subheader("Data Cleaning and Preprocessing")
    prep_cols = st.columns(4)
    prep_cols[0].metric("Rows Before", f"{prep['rows_before_cleaning']:,}")
    prep_cols[1].metric("Rows After", f"{prep['rows_after_deduplication_and_cleaning']:,}")
    prep_cols[2].metric("Duplicates Removed", f"{prep['duplicate_rows_removed']:,}")
    prep_cols[3].metric("Noise Removed", f"{prep['noisy_rows_removed']:,}")

    st.markdown("**Techniques applied**")
    st.write(
        "The preprocessing pipeline includes type conversion, missing-value imputation, duplicate removal, IQR-based noise filtering, feature fusion, MinMax scaling, Standard scaling, and PCA-based dimension reduction."
    )

    missing_left, missing_right = st.columns(2)
    missing_fig = px.bar(
        data["missing_value_report"].sort_values("missing_values_before", ascending=False).head(15),
        x="feature",
        y=["missing_values_before", "missing_values_after"],
        barmode="group",
        title="Missing Values Before and After Cleaning",
    )
    missing_left.plotly_chart(missing_fig, width="stretch")

    scaler_fig = px.bar(
        data["scaler_summary"],
        x="feature",
        y=["original_mean", "original_std"],
        barmode="group",
        title="Original Feature Scale Summary",
    )
    missing_right.plotly_chart(scaler_fig, width="stretch")

    pca_fig = px.scatter(
        data["pca_projection"].sample(min(4000, len(data["pca_projection"])), random_state=42),
        x="pca_1",
        y="pca_2",
        color="region",
        hover_data=["state", "category", "payment_type"],
        title="PCA Projection After Standard Scaling",
    )
    st.plotly_chart(pca_fig, width="stretch")

    st.markdown("**Feature fusion examples**")
    st.write(", ".join(prep["feature_fusion_examples"]))
    st.dataframe(data["scaler_summary"], width="stretch", hide_index=True)

with tab3:
    forecast_left, forecast_right = st.columns([2, 1])
    combined = data["forecast_actuals"][["date", "daily_revenue", "predicted_revenue"]].copy()
    future = data["future_forecast"].copy()
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=combined["date"], y=combined["daily_revenue"], name="Actual revenue"))
    forecast_fig.add_trace(go.Scatter(x=combined["date"], y=combined["predicted_revenue"], name="Predicted revenue"))
    forecast_fig.add_trace(go.Scatter(x=future["date"], y=future["predicted_revenue"], name="Future forecast"))
    forecast_fig.update_layout(title="Daily Sales Forecast")
    forecast_left.plotly_chart(forecast_fig, width="stretch")

    forecast_right.metric("MAE", f"{metrics['mae']:,.0f}")
    forecast_right.metric("RMSE", f"{metrics['rmse']:,.0f}")
    forecast_right.metric("MAPE", f"{metrics['mape']:.1f}%")
    forecast_right.metric("Accuracy Proxy", f"{kpis['forecast_accuracy_pct']:.1f}%")

    anomaly_fig = px.scatter(
        data["anomalies"].assign(size_value=lambda df: df["z_score"].abs().clip(lower=0.1)),
        x="date",
        y="daily_revenue",
        color="anomaly_type",
        size="size_value",
        title="Anomaly Detection on Daily Revenue",
    )
    st.plotly_chart(anomaly_fig, width="stretch")

with tab4:
    seg_left, seg_right = st.columns(2)
    seg_scope = data["product_segments"]
    if category_filter:
        seg_scope = seg_scope[seg_scope["Category"].isin(category_filter)]
    segment_filter = seg_right.multiselect(
        "Segment Filter",
        sorted(seg_scope["segment_name"].dropna().unique()),
    )
    if segment_filter:
        seg_scope = seg_scope[seg_scope["segment_name"].isin(segment_filter)]
    segment_fig = px.scatter(
        seg_scope,
        x="total_revenue",
        y="revenue_volatility",
        color="segment_name",
        size="transaction_count",
        hover_data=["product_id", "Category"],
        title="Product Segmentation",
    )
    seg_left.plotly_chart(segment_fig, width="stretch")

    customer_fig = px.bar(
        data["customer_segment_summary"],
        x="rfm_segment",
        y="revenue",
        color="customers",
        title="Customer Segment Revenue",
    )
    seg_right.plotly_chart(customer_fig, width="stretch")

    table_col1, table_col2 = st.columns([2, 1])
    search_text = table_col1.text_input("Search Product ID", placeholder="Type a product id")
    row_limit = table_col2.slider("Rows to show", min_value=10, max_value=200, value=50, step=10)

    table_scope = seg_scope.copy()
    if search_text:
        table_scope = table_scope[table_scope["product_id"].astype(str).str.contains(search_text, case=False, na=False)]

    st.dataframe(
        table_scope[
            [
                "product_id",
                "Category",
                "segment_name",
                "total_revenue",
                "total_quantity",
                "avg_unit_price",
                "transaction_count",
                "active_months",
                "avg_monthly_revenue",
                "revenue_volatility",
            ]
        ]
        .sort_values("total_revenue", ascending=False)
        .head(row_limit),
        width="stretch",
        hide_index=True,
    )

with tab5:
    st.metric("Top State", kpis["top_state"])
    st.metric("Top Category", kpis["top_category"])
    for _, row in data["recommendations"].iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['priority']} | {row['theme']}**")
            st.write(row["recommendation"])
            st.caption(row["evidence"])
