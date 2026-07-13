"""
GRU + fact-pack NLG recommendation module (academic BI + AI project).

Option D hybrid pipeline:
  1. A small GRU reads recent daily business sequences and scores action themes.
  2. A structured fact pack is built from forecasts, anomalies, segments, etc.
  3. An NLG composer turns (theme, score, facts) into dynamic sentences.

Training labels are weakly supervised heuristics (coursework-friendly). Inference
uses the GRU for theme selection; recommendation wording is composed from the
fact pack rather than a single hard-coded sentence per theme.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import MODELS_DIR


# Themes the GRU can activate. Kept small on purpose for a student project.
RECOMMENDATION_THEMES = [
    "Inventory",
    "Operations",
    "Customer Experience",
    "Portfolio",
    "Retention",
    "Expansion",
]

PRIORITY_BY_THEME = {
    "Inventory": "High",
    "Operations": "High",
    "Customer Experience": "Medium",
    "Portfolio": "Medium",
    "Retention": "Medium",
    "Expansion": "Low",
}

SEQUENCE_LENGTH = 14
HIDDEN_SIZE = 48
NUM_EPOCHS = 18
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DECISION_THRESHOLD = 0.45
RANDOM_SEED = 42


class RecommendationGRU(nn.Module):
    """Small GRU classifier used to score recommendation themes."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_classes: int = len(RECOMMENDATION_THEMES)):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.fc(self.dropout(last_hidden))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


CONTEXT_COLUMNS = [
    "ctx_review_score",
    "ctx_at_risk_customers",
    "ctx_at_risk_products",
    "ctx_top_geo_share",
]


def _safe_series_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)


def _build_daily_signal_frame(daily_sales: pd.DataFrame, anomalies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the multivariate sequence used by the GRU."""
    frame = daily_sales[["date", "daily_revenue", "daily_quantity", "avg_order_value"]].copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["rolling_7_mean"] = frame["daily_revenue"].rolling(7, min_periods=1).mean()
    frame["rolling_30_mean"] = frame["daily_revenue"].rolling(30, min_periods=1).mean()
    frame["revenue_pct_change"] = frame["daily_revenue"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    frame["revenue_vs_7d"] = _safe_series_ratio(frame["daily_revenue"], frame["rolling_7_mean"])
    frame["trend_7_30"] = _safe_series_ratio(frame["rolling_7_mean"], frame["rolling_30_mean"])

    anomaly_days = set(pd.to_datetime(anomalies["date"]).dt.normalize()) if not anomalies.empty else set()
    frame["is_anomaly"] = frame["date"].dt.normalize().isin(anomaly_days).astype(float)
    drop_days = set()
    if not anomalies.empty and "anomaly_type" in anomalies.columns:
        drop_days = set(pd.to_datetime(anomalies.loc[anomalies["anomaly_type"] == "Drop", "date"]).dt.normalize())
    frame["is_drop_anomaly"] = frame["date"].dt.normalize().isin(drop_days).astype(float)

    # Placeholder channels for static business context (filled per window).
    for column in CONTEXT_COLUMNS:
        frame[column] = 0.0

    feature_columns = [
        "daily_revenue",
        "daily_quantity",
        "avg_order_value",
        "rolling_7_mean",
        "rolling_30_mean",
        "revenue_pct_change",
        "revenue_vs_7d",
        "trend_7_30",
        "is_anomaly",
        "is_drop_anomaly",
        *CONTEXT_COLUMNS,
    ]
    return frame[["date", *feature_columns]], feature_columns


def _context_vector(
    category_performance: pd.DataFrame,
    customer_segments: pd.DataFrame,
    product_segments: pd.DataFrame,
    country_performance: pd.DataFrame,
) -> np.ndarray:
    """Static context features written on the last timestep (simple feature fusion)."""
    lowest_review = float(category_performance["avg_review_score"].min()) if not category_performance.empty else 3.0
    at_risk_customers = 0.0
    if not customer_segments.empty and "rfm_segment" in customer_segments.columns:
        total_customers = float(customer_segments["customers"].sum())
        risk_customers = float(customer_segments.loc[customer_segments["rfm_segment"] == "At risk", "customers"].sum())
        at_risk_customers = _safe_ratio(risk_customers, total_customers)

    at_risk_products = 0.0
    if not product_segments.empty and "segment_name" in product_segments.columns:
        at_risk_products = _safe_ratio(
            (product_segments["segment_name"] == "At-risk products").sum(),
            len(product_segments),
        )

    top_share = 0.0
    if not country_performance.empty and "revenue" in country_performance.columns:
        total_rev = float(country_performance["revenue"].sum())
        top_rev = float(country_performance["revenue"].max())
        top_share = _safe_ratio(top_rev, total_rev)

    return np.array([lowest_review / 5.0, at_risk_customers, at_risk_products, top_share], dtype=np.float32)


def _heuristic_labels_for_window(
    window: pd.DataFrame,
    future_slice: pd.DataFrame | None,
    context: np.ndarray,
) -> np.ndarray:
    """
    Weak labels used only during training.

    These give the GRU something to learn from. Inference uses the trained
    network rather than these rules directly.
    """
    labels = np.zeros(len(RECOMMENDATION_THEMES), dtype=np.float32)

    recent_rev = float(window["daily_revenue"].sum())
    if future_slice is not None and len(future_slice) > 0:
        future_rev = float(future_slice["daily_revenue"].sum())
    else:
        # Fallback proxy used near the end of history.
        future_rev = float(window["daily_revenue"].tail(7).mean() * len(window))

    growth = _safe_ratio(future_rev - recent_rev, recent_rev) * 100
    drop_count = float(window["is_drop_anomaly"].sum())
    lowest_review, at_risk_customers, at_risk_products, top_share = context.tolist()

    if growth > 8:
        labels[0] = 1.0  # Inventory
    if drop_count >= 2:
        labels[1] = 1.0  # Operations
    if lowest_review < 0.78:  # roughly review < 3.9
        labels[2] = 1.0  # Customer Experience
    if at_risk_products > 0.2:
        labels[3] = 1.0  # Portfolio
    if at_risk_customers > 0.15:
        labels[4] = 1.0  # Retention
    if top_share > 0.25:
        labels[5] = 1.0  # Expansion

    # Always keep at least one positive label so training stays stable.
    if labels.sum() == 0:
        labels[5] = 1.0

    return labels


def _inject_context(window_vals: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Copy static context into the reserved channels of the last day only."""
    enriched = window_vals.copy()
    enriched[:, -4:] = 0.0
    enriched[-1, -4:] = context
    return enriched


def _create_training_windows(
    scaled_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_columns: list[str],
    context: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    # Features are scaled for the network; labels stay on raw revenue values.
    values = scaled_frame[feature_columns].to_numpy(dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for end_idx in range(seq_len, len(scaled_frame)):
        start_idx = end_idx - seq_len
        window_vals = _inject_context(values[start_idx:end_idx], context)

        future_end = min(end_idx + seq_len, len(raw_frame))
        future_slice = raw_frame.iloc[end_idx:future_end]
        window_df = raw_frame.iloc[start_idx:end_idx]
        label = _heuristic_labels_for_window(window_df, future_slice, context)

        xs.append(window_vals)
        ys.append(label)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_recommendation_gru(
    daily_sales: pd.DataFrame,
    anomalies: pd.DataFrame,
    category_performance: pd.DataFrame,
    customer_segments: pd.DataFrame,
    product_segments: pd.DataFrame,
    country_performance: pd.DataFrame,
) -> tuple[RecommendationGRU, StandardScaler, dict[str, float], list[str]]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    raw_frame, feature_columns = _build_daily_signal_frame(daily_sales, anomalies)
    context = _context_vector(category_performance, customer_segments, product_segments, country_performance)

    # Scale continuous signals (skip binary / context channels that are already 0-1-ish).
    continuous_cols = [
        c
        for c in feature_columns
        if c not in {"is_anomaly", "is_drop_anomaly", *CONTEXT_COLUMNS}
    ]
    scaler = StandardScaler()
    scaled_frame = raw_frame.copy()
    scaled_frame[continuous_cols] = scaler.fit_transform(raw_frame[continuous_cols])

    x_data, y_data = _create_training_windows(scaled_frame, raw_frame, feature_columns, context)
    if len(x_data) < 20:
        raise RuntimeError("Not enough daily history to train the recommendation GRU.")

    split_idx = int(len(x_data) * 0.8)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = RecommendationGRU(input_size=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(NUM_EPOCHS):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val))
        val_probs = torch.sigmoid(val_logits).numpy()
        val_pred = (val_probs >= DECISION_THRESHOLD).astype(int)
        # Micro F1 is enough for our multi-label coursework metric.
        f1 = float(f1_score(y_val.astype(int), val_pred, average="micro", zero_division=0))

    metrics = {
        "recommendation_f1_micro": f1,
        "train_windows": float(len(x_train)),
        "val_windows": float(len(x_val)),
        "sequence_length": float(SEQUENCE_LENGTH),
        "hidden_size": float(HIDDEN_SIZE),
        "epochs": float(NUM_EPOCHS),
    }
    return model, scaler, metrics, feature_columns


def _latest_sequence(
    daily_sales: pd.DataFrame,
    anomalies: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
    context: np.ndarray,
) -> np.ndarray:
    signal_frame, _ = _build_daily_signal_frame(daily_sales, anomalies)
    continuous_cols = [
        c
        for c in feature_columns
        if c not in {"is_anomaly", "is_drop_anomaly", *CONTEXT_COLUMNS}
    ]
    signal_frame[continuous_cols] = scaler.transform(signal_frame[continuous_cols])

    window = signal_frame[feature_columns].tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
    if len(window) < SEQUENCE_LENGTH:
        pad = np.repeat(window[:1], SEQUENCE_LENGTH - len(window), axis=0)
        window = np.vstack([pad, window])
    return _inject_context(window, context)


def _money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "urgent"
    if confidence >= 0.65:
        return "elevated"
    return "watch"


def _dynamic_priority(theme: str, confidence: float, facts: dict) -> str:
    """Priority is not fixed forever: confidence + fact pressure can raise/lower it."""
    base = PRIORITY_BY_THEME[theme]
    severity = _severity_from_confidence(confidence)

    pressure = 0
    if theme == "Inventory" and facts["forecast_growth_pct"] > 15:
        pressure += 1
    if theme == "Operations" and facts["drop_anomaly_count"] >= 8:
        pressure += 1
    if theme == "Customer Experience" and facts["weakest_category_score"] < 3.0:
        pressure += 1
    if theme == "Retention" and facts["at_risk_customer_share"] > 0.2:
        pressure += 1
    if theme == "Portfolio" and facts["at_risk_product_share"] > 0.35:
        pressure += 1

    if severity == "urgent" and pressure >= 1:
        return "High"
    if base == "High" and severity == "watch" and pressure == 0:
        return "Medium"
    if base == "Low" and severity == "urgent":
        return "Medium"
    return base


def build_fact_pack(
    kpis: dict[str, float | int | str],
    future_forecast: pd.DataFrame,
    recent_daily_sales: pd.DataFrame,
    anomalies: pd.DataFrame,
    product_segments: pd.DataFrame,
    category_performance: pd.DataFrame,
    customer_segments: pd.DataFrame,
    country_performance: pd.DataFrame,
) -> dict:
    """
    Structured evidence bundle used by the NLG layer.

    The GRU only picks themes + scores. Sentence text is composed later from
    these facts so comments stay data-driven instead of hard-coded phrases.
    """
    recent_30 = float(recent_daily_sales.tail(30)["daily_revenue"].sum())
    recent_7 = float(recent_daily_sales.tail(7)["daily_revenue"].sum())
    prev_7 = float(recent_daily_sales.tail(14).head(7)["daily_revenue"].sum()) if len(recent_daily_sales) >= 14 else recent_7
    next_30 = float(future_forecast["predicted_revenue"].sum())
    growth = _safe_ratio(next_30 - recent_30, recent_30) * 100
    week_change = _safe_ratio(recent_7 - prev_7, prev_7) * 100

    drop_count = int((anomalies["anomaly_type"] == "Drop").sum()) if not anomalies.empty and "anomaly_type" in anomalies.columns else 0
    spike_count = int((anomalies["anomaly_type"] == "Spike").sum()) if not anomalies.empty and "anomaly_type" in anomalies.columns else 0
    anomaly_total = int(len(anomalies)) if anomalies is not None else 0

    weakest = category_performance.sort_values(["avg_review_score", "revenue"], ascending=[True, False]).iloc[0]
    strongest = category_performance.sort_values(["avg_review_score", "revenue"], ascending=[False, False]).iloc[0]
    top_category = category_performance.sort_values("revenue", ascending=False).iloc[0]

    at_risk_products = product_segments[product_segments["segment_name"] == "At-risk products"] if not product_segments.empty else product_segments
    high_products = product_segments[product_segments["segment_name"] == "High performers"] if not product_segments.empty else product_segments
    example_ids = (
        ", ".join(at_risk_products.sort_values("total_revenue").head(3)["product_id"].astype(str).tolist())
        if not at_risk_products.empty
        else "n/a"
    )
    top_product_rev = float(high_products["total_revenue"].max()) if not high_products.empty else 0.0

    at_risk_customers = customer_segments[customer_segments["rfm_segment"] == "At risk"] if not customer_segments.empty else customer_segments
    champions = customer_segments[customer_segments["rfm_segment"] == "Champions"] if not customer_segments.empty else customer_segments
    total_customers = float(customer_segments["customers"].sum()) if not customer_segments.empty else 0.0
    at_risk_n = int(at_risk_customers["customers"].sum()) if not at_risk_customers.empty else 0
    champions_n = int(champions["customers"].sum()) if not champions.empty else 0

    top_geo = country_performance.sort_values("revenue", ascending=False).iloc[0]
    geo_col = "Country" if "Country" in top_geo.index else ("state" if "state" in top_geo.index else country_performance.columns[0])
    total_geo_rev = float(country_performance["revenue"].sum())
    top_geo_rev = float(top_geo["revenue"])
    second_geo_rev = float(country_performance.sort_values("revenue", ascending=False).iloc[1]["revenue"]) if len(country_performance) > 1 else 0.0

    return {
        "recent_30d_revenue": recent_30,
        "forecast_30d_revenue": next_30,
        "forecast_growth_pct": growth,
        "week_over_week_pct": week_change,
        "drop_anomaly_count": drop_count,
        "spike_anomaly_count": spike_count,
        "anomaly_total": anomaly_total,
        "weakest_category": str(weakest["Category"]),
        "weakest_category_score": float(weakest["avg_review_score"]),
        "strongest_category": str(strongest["Category"]),
        "strongest_category_score": float(strongest["avg_review_score"]),
        "top_category": str(top_category["Category"]),
        "top_category_revenue": float(top_category["revenue"]),
        "at_risk_product_count": int(len(at_risk_products)),
        "at_risk_product_share": _safe_ratio(len(at_risk_products), max(len(product_segments), 1)),
        "example_at_risk_product_ids": example_ids,
        "top_product_revenue": top_product_rev,
        "at_risk_customers": at_risk_n,
        "at_risk_customer_share": _safe_ratio(at_risk_n, total_customers),
        "champions_customers": champions_n,
        "top_geography": str(top_geo[geo_col]),
        "top_geography_revenue": top_geo_rev,
        "top_geography_share": _safe_ratio(top_geo_rev, total_geo_rev),
        "second_geography_revenue": second_geo_rev,
        "avg_review": float(category_performance["avg_review_score"].mean()) if not category_performance.empty else 0.0,
        "late_order_rate": float(kpis.get("late_order_rate_pct", 0) or 0),
        "top_state_kpi": str(kpis.get("top_state", "")),
        "top_category_kpi": str(kpis.get("top_category", "")),
    }


def compose_recommendation_text(theme: str, confidence: float, facts: dict) -> tuple[str, str, str]:
    """
    Lightweight NLG: assemble action + evidence sentences from fact slots.

    Wording changes with severity and live metrics (not one fixed phrase per theme).
    """
    severity = _severity_from_confidence(confidence)
    priority = _dynamic_priority(theme, confidence, facts)
    conf_txt = f"{confidence:.0%}"

    if theme == "Inventory":
        growth = facts["forecast_growth_pct"]
        horizon_rev = _money(facts["forecast_30d_revenue"])
        wow = facts["week_over_week_pct"]
        if growth > 12 and severity == "urgent":
            action = (
                f"Immediately lift stock cover on {facts['top_category']} and other high movers — "
                f"demand for the next 30 days is projected at {horizon_rev} ({growth:+.1f}% vs the last 30 days)."
            )
        elif growth > 5:
            action = (
                f"Increase replenishment depth for top SKUs ahead of a {growth:+.1f}% forecast lift "
                f"({horizon_rev} expected). Week-over-week sales are currently {wow:+.1f}%."
            )
        elif growth < -5:
            action = (
                f"Tighten intake and freeze slow movers: the 30-day outlook is {growth:+.1f}% "
                f"({horizon_rev}), so avoid overstocking weak categories."
            )
        else:
            action = (
                f"Keep inventory aligned with a near-flat outlook ({growth:+.1f}%, {horizon_rev}); "
                f"protect availability on {facts['top_category']} while trimming excess elsewhere."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: last-30d actual {_money(facts['recent_30d_revenue'])}, "
            f"forecast {_money(facts['forecast_30d_revenue'])}, WoW {wow:+.1f}%."
        )

    elif theme == "Operations":
        drops = facts["drop_anomaly_count"]
        spikes = facts["spike_anomaly_count"]
        late_txt = f"{facts['late_order_rate']:.1f}%"

        if drops >= 8 or severity == "urgent":
            action = (
                f"Run an urgent fulfilment audit: {drops} abnormal drop days and {spikes} spikes "
                f"were flagged, with late orders at {late_txt} — focus first on {facts['top_geography']}."
            )
        elif drops >= 3:
            action = (
                f"Investigate stock-outs and carrier delays linked to {drops} low-sales anomaly days "
                f"(plus {spikes} spikes). Prioritise ops checks in {facts['top_geography']}."
            )
        else:
            action = (
                f"Monitor ops volatility ({drops} drops / {spikes} spikes detected) and stress-test "
                f"handoffs in {facts['top_geography']} where revenue concentration is highest."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: {facts['anomaly_total']} anomaly days total, late-order signal {late_txt}, "
            f"geo focus {facts['top_geography']}."
        )

    elif theme == "Customer Experience":
        weak = facts["weakest_category"]
        weak_score = facts["weakest_category_score"]
        strong = facts["strongest_category"]
        gap = facts["strongest_category_score"] - weak_score
        if weak_score < 3.0 or severity == "urgent":
            action = (
                f"Launch a recovery plan for {weak} (avg review {weak_score:.2f}/5): "
                f"quality checks, clearer delivery promises, and post-purchase follow-up. "
                f"Benchmark against {strong} ({facts['strongest_category_score']:.2f}/5)."
            )
        elif weak_score < 3.8:
            action = (
                f"Improve review drivers for {weak} ({weak_score:.2f}/5) — the gap to {strong} "
                f"is {gap:.2f} points. Start with packaging, SLA messaging, and complaint tagging."
            )
        else:
            action = (
                f"Protect CX on {weak} while reviews remain soft at {weak_score:.2f}/5; "
                f"reuse practices from {strong} ({facts['strongest_category_score']:.2f}/5) to close the gap."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: weakest={weak} ({weak_score:.2f}), strongest={strong} "
            f"({facts['strongest_category_score']:.2f}), catalogue leader={facts['top_category']}."
        )

    elif theme == "Portfolio":
        n_risk = facts["at_risk_product_count"]
        share = facts["at_risk_product_share"] * 100
        ids = facts["example_at_risk_product_ids"]
        if share > 35 or severity == "urgent":
            action = (
                f"Aggressive assortments clean-up: {n_risk} at-risk products ({share:.0f}% of SKUs). "
                f"Bundle, discount, or delist the weakest set (e.g. {ids}) within the next planning cycle."
            )
        elif n_risk > 0:
            action = (
                f"Rebalance the catalogue around {n_risk} at-risk SKUs ({share:.0f}% share). "
                f"Test promotions/bundles on IDs such as {ids}, and shift shelf space to high performers "
                f"(top product revenue {_money(facts['top_product_revenue'])})."
            )
        else:
            action = (
                f"Portfolio risk is currently light; keep nurturing high performers "
                f"(peak product revenue {_money(facts['top_product_revenue'])}) and watch for new at-risk clusters."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: at-risk SKUs={n_risk} ({share:.0f}%), examples={ids}."
        )

    elif theme == "Retention":
        n_risk = facts["at_risk_customers"]
        share = facts["at_risk_customer_share"] * 100
        champs = facts["champions_customers"]
        if share > 20 or severity == "urgent":
            action = (
                f"Trigger a win-back programme for {n_risk:,} at-risk customers ({share:.1f}% of the base). "
                f"Personalise offers using Champion behaviours ({champs:,} Champions) and focus on recent dormancy."
            )
        elif n_risk > 0:
            action = (
                f"Design a targeted reactivation wave covering {n_risk:,} at-risk shoppers ({share:.1f}%). "
                f"Start with mid-value dormant buyers, then protect the {champs:,} Champions with loyalty perks."
            )
        else:
            action = (
                f"Retention pressure is limited right now; reinforce loyalty touches for the {champs:,} Champions "
                f"and watch early dormancy signals."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: at-risk={n_risk:,} ({share:.1f}%), Champions={champs:,}."
        )

    else:  # Expansion
        geo = facts["top_geography"]
        share = facts["top_geography_share"] * 100
        geo_rev = _money(facts["top_geography_revenue"])
        second = _money(facts["second_geography_revenue"])
        if share > 40 or severity == "urgent":
            action = (
                f"Treat {geo} as the playbook market ({share:.0f}% of revenue, {geo_rev}): "
                f"replicate assortment depth and service SLAs there, then transfer winning patterns "
                f"to the next largest geography ({second})."
            )
        else:
            action = (
                f"Use {geo} ({share:.0f}% / {geo_rev}) as the benchmark for expansion tests, "
                f"and compare conversion/fulfilment quality against the runner-up market ({second})."
            )
        evidence = (
            f"GRU theme score {conf_txt} ({severity}). "
            f"Fact pack: top geo={geo} ({geo_rev}, {share:.0f}% share), "
            f"KPI top state={facts['top_state_kpi'] or geo}."
        )

    return action, evidence, priority


def _render_recommendation(theme: str, confidence: float, facts: dict) -> dict[str, str | float]:
    """Hybrid step: GRU theme/score + fact pack → Groq LLM wording (fallback: local NLG)."""
    from .llm_nlg import generate_recommendation_copy

    action, evidence, priority = compose_recommendation_text(theme, confidence, facts)
    severity = _severity_from_confidence(confidence)
    text_source = "fact_pack_nlg"
    model_label = "GRU+NLG"

    llm_copy = generate_recommendation_copy(
        theme=theme,
        confidence=confidence,
        severity=severity,
        priority=priority,
        facts=facts,
    )
    if llm_copy:
        action = llm_copy["recommendation"]
        evidence = llm_copy["evidence"]
        text_source = "groq_llm"
        model_label = f"GRU+Groq/{llm_copy.get('model', 'llm')}"

    return {
        "priority": priority,
        "theme": theme,
        "recommendation": action,
        "evidence": evidence,
        "model_confidence": round(float(confidence), 4),
        "model": model_label,
        "severity": severity,
        "text_source": text_source,
    }


def build_recommendations(
    kpis: dict[str, float | int | str],
    future_forecast: pd.DataFrame,
    recent_daily_sales: pd.DataFrame,
    anomalies: pd.DataFrame,
    product_segments: pd.DataFrame,
    category_performance: pd.DataFrame,
    customer_segments: pd.DataFrame,
    country_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Option D hybrid pipeline:
      1) train/score GRU themes
      2) build a structured fact pack from live analytics
      3) compose recommendation sentences with NLG over that fact pack
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model, scaler, metrics, feature_columns = train_recommendation_gru(
        daily_sales=recent_daily_sales,
        anomalies=anomalies,
        category_performance=category_performance,
        customer_segments=customer_segments,
        product_segments=product_segments,
        country_performance=country_performance,
    )

    context = _context_vector(category_performance, customer_segments, product_segments, country_performance)
    latest = _latest_sequence(recent_daily_sales, anomalies, scaler, feature_columns, context)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(latest[None, ...]))
        probs = torch.sigmoid(logits).numpy()[0]

    activated = [
        (theme, float(prob))
        for theme, prob in zip(RECOMMENDATION_THEMES, probs)
        if prob >= DECISION_THRESHOLD
    ]
    # If the model is unsure, keep the top-scoring themes so the dashboard is never empty.
    if not activated:
        top_idx = np.argsort(probs)[::-1][:3]
        activated = [(RECOMMENDATION_THEMES[i], float(probs[i])) for i in top_idx]

    activated = sorted(activated, key=lambda item: item[1], reverse=True)

    facts = build_fact_pack(
        kpis=kpis,
        future_forecast=future_forecast,
        recent_daily_sales=recent_daily_sales,
        anomalies=anomalies,
        product_segments=product_segments,
        category_performance=category_performance,
        customer_segments=customer_segments,
        country_performance=country_performance,
    )
    (MODELS_DIR / "recommendation_fact_pack.json").write_text(
        json.dumps(facts, indent=2),
        encoding="utf-8",
    )

    rows = [_render_recommendation(theme, confidence, facts) for theme, confidence in activated]

    artifact = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "feature_columns": feature_columns,
        "themes": RECOMMENDATION_THEMES,
        "metrics": metrics,
        "threshold": DECISION_THRESHOLD,
        "pipeline": "GRU+fact_pack_NLG",
    }
    joblib.dump(artifact, MODELS_DIR / "recommendation_gru.joblib")
    torch.save(model.state_dict(), MODELS_DIR / "recommendation_gru.pt")

    (MODELS_DIR / "recommendation_gru_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return pd.DataFrame(rows)
