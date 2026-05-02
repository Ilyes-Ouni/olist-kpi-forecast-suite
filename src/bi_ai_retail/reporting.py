from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_markdown_report(path: Path, summary: dict, model_metrics: dict, recommendations: pd.DataFrame) -> None:
    lines = [
        "# Olist Commerce Decision Lab",
        "",
        "## Executive Summary",
        f"- Total revenue: ${summary['total_revenue']:,.2f}",
        f"- Total profit: ${summary['total_profit']:,.2f}",
        f"- Gross margin: {summary['gross_margin_pct']:.2f}%",
        f"- Orders analysed: {summary['orders_count']:,}",
        f"- Customers analysed: {summary['customers_count']:,}",
        f"- Repeat customer rate: {summary['repeat_customer_rate']:.1f}%",
        f"- Top state: {summary['top_state']}",
        f"- Top category: {summary['top_category']}",
        f"- Forecasted next 30 days revenue: ${summary['forecast_30d_revenue']:,.2f}",
        f"- Rows after cleaning: {summary['rows_after_cleaning']:,}",
        f"- Noisy rows removed: {summary['noisy_rows_removed']:,}",
        "",
        "## Forecasting Performance",
        f"- MAE: {model_metrics['mae']:.2f}",
        f"- RMSE: {model_metrics['rmse']:.2f}",
        f"- MAPE: {model_metrics['mape']:.2f}%",
        f"- Forecast accuracy proxy: {summary['forecast_accuracy_pct']:.2f}%",
        "",
        "## Management Actions",
    ]

    for _, row in recommendations.iterrows():
        lines.append(f"- [{row['priority']}] {row['theme']}: {row['recommendation']} ({row['evidence']})")

    path.write_text("\n".join(lines), encoding="utf-8")
