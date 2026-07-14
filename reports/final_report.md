# Final Report

## Project title

Olist Commerce Decision Lab

## Team roles (suggested split for a 2-student project)


| Focus                           | Typical ownership                                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Student A — BI & product**    | Data preparation story, KPI design, Streamlit Executive / Preparation / Segmentation tabs, business narrative, demo BI walkthrough                                  |
| **Student B — AI & evaluation** | Forecasting comparison (GBR / XGBoost / Prophet), Isolation Forest anomalies, GRU + fact-pack + Groq recommendations, metrics discussion, methodology & limitations |


Both students should be able to answer high-level questions on the full pipeline.

## Problem statement

Retail organisations usually monitor past sales through descriptive dashboards, but they still struggle to anticipate future demand and quickly react to abnormal sales patterns. This project addresses that gap by combining Business Intelligence and Artificial Intelligence in one decision-support platform built on the Brazilian **Olist** e-commerce dataset.

## Objectives

1. Prepare a clean retail analytics dataset (typing, imputation, IQR noise filtering, feature fusion, scaling, PCA).
2. Build KPI tables and business views for revenue, profit, geography, and categories.
3. Forecast short-term daily demand using a **multi-model** workflow and an explicit selection rule.
4. Detect unusual sales spikes and drops (Isolation Forest + Z-score).
5. Segment customers (RFM) and products (K-Means).
6. Generate dynamic recommendations (GRU theme scores → structured fact pack → Groq LLM / local NLG).
7. Deliver an interactive Streamlit dashboard for decision-makers and academic demonstration.

## Methodology

### Data preparation

- Type conversion for dates, numeric, and categorical fields
- Missing-value treatment (median / default category)
- Duplicate removal and IQR-based noise filtering
- Engineered features (volume, freight ratios, delivery delay, etc.)
- MinMax + Standard scaling and PCA projection for the preparation tab

### BI analytics

- Executive KPIs (revenue, orders, customers, reviews, late orders, 30-day forecast)
- State / region and category performance
- Monthly trend analysis
- Interactive filters in the dashboard

### AI analytics

#### Forecasting (regression)

Candidates trained on the **same** hold-out split (`train < 2018-05-01`, test afterwards):

1. **Gradient Boosting** (sklearn) with lag / calendar / rolling features
2. **XGBoost** with the same feature set
3. **Prophet** (trend + weekly/yearly seasonality)

Metrics computed for every candidate: **MAE, MSE, RMSE, MAPE, R²**.

**Selection rule:** lowest RMSE, then lowest MAPE. Only the winner produces the official next-30-day forecast used in KPIs and recommendation facts.

*Interpretation tip:* on this Olist daily series, lag-based tree models typically beat Prophet because short-term autocorrelation matters more than a smooth seasonal skeleton.

Isolation Forest is **not** in this table — it does not predict revenue; it detects outliers.

#### Anomaly detection

- Isolation Forest on multivariate daily signals (`contamination=0.03`)
- Z-score magnitude filter (`|z| ≥ 2.5`)
- Labels: Spike vs Drop relative to a rolling mean  
- Dashboard shows spike/drop counts and anomaly rate (% of days)

#### Segmentation

- Customer RFM segments (Champions, Loyal, At risk, …)
- Product K-Means clusters (High performers, Seasonal opportunities, At-risk products)

#### Recommendations (hybrid Option D)

1. **GRU** scores action themes from 14-day business sequences (weak labels used only for training — coursework-friendly supervised signal).
2. A **fact pack** extracts live evidence (growth, anomalies, weak categories, at-risk customers/SKUs, top geography).
3. **Groq** (`llama-3.3-70b-versatile`) writes action/evidence sentences from that fact pack.
4. If the API key is missing, **local NLG** falls back so the demo still works offline.

## Evaluation summary


| Layer           | What we evaluate       | Main indicators                               |
| --------------- | ---------------------- | --------------------------------------------- |
| Forecast        | Hold-out daily revenue | MAE, MSE, RMSE, MAPE, R² + residual histogram |
| Anomalies       | Unusual days           | Count, spike/drop split, rate vs calendar     |
| Recommendations | Theme ranking quality  | GRU micro-F1 (+ qualitative evidence quotes)  |


## Limitations (critical discussion)

- Historical window ends in **2018** — patterns may not transfer to today’s market.
- Tree forecasts are **recursive** for multi-step horizons → error can accumulate.
- Prophet assumptions (additive seasonality) may underfit volatile demand.
- GRU labels are **weakly supervised** heuristics — honest academic limitation, not pure human-labelled actions.
- Groq wording depends on the fact pack and API availability; we never put API keys in git.
- Product K-Means labels are post-hoc business names; cluster meaning should be validated with domain experts.

## Deliverables

- Runnable Python pipeline (`main.py`)
- Interactive Streamlit dashboard (`app.py`)
- Processed CSV / JSON outputs (`data/processed/`)
- Model comparison artefacts (`model_comparison.csv`, residual view in UI)
- Final report, presentation outline, **live demo script**
- GitHub repository

## Conclusion

Olist Commerce Decision Lab shows how BI and AI complement each other in a realistic e-commerce setting. BI explains what happened; the AI layer adds forecast selection, anomaly awareness, segmentation, and actionable recommendations — with explicit metrics and known limitations suitable for an academic defence.