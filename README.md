# Olist Commerce Decision Lab

Olist Commerce Decision Lab is a BI + AI academic project that turns Brazilian **Olist** e-commerce data into an interactive decision-support system: KPIs and exploration, multi-model demand forecasting, anomaly detection, customer/product segmentation, and hybrid AI recommendations.

**GitHub:** [Ilyes-Ouni/olist-kpi-forecast-suite](https://github.com/Ilyes-Ouni/olist-kpi-forecast-suite)

```text
https://github.com/Ilyes-Ouni/olist-kpi-forecast-suite.git
```

## What the platform answers

- Which categories and states drive revenue?
- How do delivery performance and reviews affect quality?
- What revenue can we expect in the next 30 days — and **which model** should we trust?
- Which daily sales patterns look abnormal?
- Which customers / products need action?
- What should management do next (prioritised recommendations)?

## Architecture (at a glance)

1. **Data prep** — typing, imputation, IQR noise filter, feature fusion, MinMax / Standard scaling, PCA  
2. **BI layer** — KPIs, geography, categories, monthly trends, Streamlit filters  
3. **Forecasting** — Gradient Boosting vs XGBoost vs Prophet → select by **RMSE then MAPE**  
4. **Anomalies** — Isolation Forest + Z-score (Spike / Drop) — *not* a forecast candidate  
5. **Segmentation** — RFM customers + K-Means products  
6. **Recommendations** — GRU theme scores → fact pack → Groq LLM (fallback: local NLG)  
7. **Dashboard** — Streamlit tabs for live academic demonstration  

## Team roles (2 students)

| Role | Focus |
| --- | --- |
| **Student A — BI & product** | Preprocessing narrative, KPIs, Executive / Preparation / Segmentation tabs, business storytelling |
| **Student B — AI & evaluation** | Forecast comparison, residuals, anomalies, GRU + Groq recommendations, metrics & limitations defence |

Live click-path and spoken lines: [`presentation/demo_script.md`](presentation/demo_script.md)

## Dataset

- Source: [Olist merged e-commerce dataset (Hugging Face)](https://huggingface.co/datasets/abhimlv/Olist-preprocessed-data-merged/resolve/main/orders_final_merged_prepro_and_feature_engg.csv)
- Type: Brazilian e-commerce transactions  
- Period: `2016-10-03` → `2018-08-29`

## Tools

Python · Pandas · NumPy · scikit-learn · XGBoost · Prophet · PyTorch · Plotly · Streamlit · Joblib · OpenAI SDK (Groq-compatible) · python-pptx

## Project structure

```text
olist-kpi-forecast-suite/
|-- app.py                 # Streamlit dashboard
|-- main.py                # Run full pipeline
|-- requirements.txt
|-- .env.example           # Groq key template (never commit real .env)
|-- presentation/
|   |-- demo_script.md     # Exact demo talk track
|   |-- presentation_outline.md
|   `-- generate_presentation.py
|-- reports/
|   |-- final_report.md
|   `-- project_summary.md
`-- src/bi_ai_retail/
    |-- data_pipeline.py
    |-- modeling.py        # Forecast candidates + anomalies + clustering
    |-- recommendations.py # GRU + fact pack + NLG
    `-- llm_nlg.py         # Groq wording helper
```

## How to run (reproducible)

### 1. Clone and install

```powershell
git clone https://github.com/Ilyes-Ouni/olist-kpi-forecast-suite.git
cd olist-kpi-forecast-suite
py -3.14 -m pip install -r requirements.txt
```

### 2. Optional: Groq wording (recommendations)

```powershell
copy .env.example .env
# Edit .env and set GROQ_API_KEY=...
# Optional: GROQ_MODEL=llama-3.3-70b-versatile
```

Without a key, recommendations still work via **local fact-pack NLG**.

### 3. Build analytics outputs

```powershell
py -3.14 main.py
```

First run downloads the dataset (may take a few minutes) and writes `data/processed/`.

### 4. Launch the dashboard

```powershell
py -3.14 -m streamlit run app.py
```

Open **http://localhost:8501**. If cards look outdated: **☰ → Clear cache → Rerun**.

## Forecasting model selection

| Candidate | Role |
| --- | --- |
| Gradient Boosting | Sklearn baseline with lag / rolling features |
| XGBoost | Strong tree competitor on the same features |
| Prophet | Classical trend + seasonality baseline |

**Metrics:** MAE, MSE, RMSE, MAPE, R² (same hold-out for all).  
**Rule:** lowest **RMSE**, then **MAPE**.  
**Winner only** writes the official next-30-day forecast (KPIs + recommendation facts).

Dashboard extras: candidate overlay chart, residual histogram, “why winner” note.

Isolation Forest is for **anomaly detection**, not this comparison table.

## Anomaly detection

- Isolation Forest (`contamination=0.03`) + Z-score (`|z| ≥ 2.5`)
- Spike vs Drop labels  
- UI shows counts, rate (% of days), and scatter plot  

## Recommendation logic

Hybrid **Option D** pipeline:

1. GRU scores themes from 14-day sequences (weak labels only for training).  
2. Fact pack aggregates live evidence.  
3. Groq `llama-3.3-70b-versatile` composes action/evidence text (fallback: local NLG).  

Secrets stay in gitignored `.env` (see `.env.example`).

## Main outputs

Under `data/processed/` (local; gitignored):

- `clean_retail.csv`, KPI JSON, segment tables  
- `forecast_actuals.csv`, `future_forecast.csv`  
- `model_comparison.csv`, `forecast_candidate_actuals.csv`  
- `anomalies.csv`, `recommendations.csv`  
- preprocessing / PCA artefacts  

## Results summary (reference run)

- Revenue: `$11.46M` · Profit: `$2.26M` · Gross margin: `19.7%`  
- Orders: `80,168` · Customers: `77,775` · States: `27`  
- Repeat customer rate: `2.8%`  
- Hold-out winner typically **Gradient Boosting** (~RMSE 6.0k, MAPE ~26%)  
- Anomaly days: ~`20`  
- Rows after cleaning: `95,351` (noisy rows removed: `17,839`)  

## Limitations (be ready to discuss)

- Dataset ends in 2018 → limited external validity  
- Recursive multi-step forecasts can accumulate error  
- Prophet may underfit volatile daily demand  
- GRU uses weakly supervised training labels  
- LLM wording needs fact constraints + offline fallback  
- K-Means business labels are interpretive  

## Documentation for marking / viva

- Full write-up: [`reports/final_report.md`](reports/final_report.md)  
- **Rapport PFE LaTeX (style TEK-UP):** [`reports/pfe_tekup/`](reports/pfe_tekup/) — auteurs Ilyes El Ouni & Montassar Zouaghi  
- Slide outline: [`presentation/presentation_outline.md`](presentation/presentation_outline.md)  
- **Exact demo script:** [`presentation/demo_script.md`](presentation/demo_script.md)  

## Academic skills demonstrated

Data cleaning · feature engineering · KPI design · BI storytelling · multi-model forecasting & selection · anomaly detection · segmentation · sequence model + LLM hybrid recommendations · interactive dashboard · critical limitations analysis
