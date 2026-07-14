# Live demonstration script

Exact talk track for the academic demo (~8–12 minutes).  
Swap **Student A / Student B** names as needed.

---

## Before you start (2 minutes, offline)

- [ ] Laptop charged; browser ready; phone on silent
- [ ] `.env` has `GROQ_API_KEY` **or** accept local NLG fallback (say this aloud if offline)
- [ ] Run once if needed:
  - `py -3.14 -m pip install -r requirements.txt`
  - `py -3.14 main.py` *(only if `data/processed` is missing)*
  - `py -3.14 -m streamlit run app.py`
- [ ] Open **http://localhost:8501**
- [ ] Clear Streamlit cache if wording/metrics look stale (**☰ → Clear cache → Rerun**)
- [ ] Decide who clicks (usually Student A) and who narrates technical slides (Student B)

---

## Opening (30–45 seconds) — both standing

**Say:**
- “We present **Olist Commerce Decision Lab**: a BI + AI decision-support system on Brazilian e-commerce data.”
- “BI answers *what happened*; AI answers *what may happen next* and *what to do*.”
- “Today we show the live dashboard, then defend how we chose the forecast model and built recommendations.”

**Do:** stay on the homepage / Executive BI tab; pointer ready.

---

## 1) Problem & value (45 seconds)

**Say:**
- “Managers usually get static past reports.”
- “They still miss: next-month demand, abnormal sales days, and prioritised actions.”
- “Our platform closes that loop in one place.”

**Do:** no clicks yet — keep KPI strip visible (Revenue, Orders, Customers, Forecast).

---

## 2) Executive BI walkthrough — Student A (90 seconds)

**Do:**
- Stay on **Executive BI**
- Point at the top KPI cards
- Optionally set a **State** or **Category** filter, then clear it again to show interactivity

**Say:**
- “These are live KPIs from cleaned Olist orders: about **$11.5M** revenue, **80k** orders, **78k** customers.”
- “Late-order rate and average review track operational quality, not only sales.”
- “Filters let a manager slice by region, category, or state without rebuilding reports.”
- “Next 30 days forecast feeds inventory and planning discussions.”

---

## 3) Data preparation — Student A (60 seconds)

**Do:** open **Data Preparation** tab; scroll PCA / missing / scaler summary briefly.

**Say:**
- “We didn’t start from a clean BI cube: we typed fields, imputed missings, removed duplicates, and applied IQR noise filtering.”
- “We engineered retail features (delays, freight ratios, volumes) then applied MinMax, Standard scaling, and PCA.”
- “This tab proves preprocessing for the academic marking scheme — feature fusion, scaling, dimension reduction.”

---

## 4) Forecasting & model selection — Student B (2–2.5 minutes) ⭐ critical

**Do:**
- Open **Forecasting**
- Point to the **comparison table** first
- Then the main forecast chart (actual vs selected vs dotted candidates)
- Then residual histogram
- Then selected-model metrics on the right

**Say:**
- “Forecasting is a **regression** task: we predict daily revenue.”
- “We trained **three candidates on the same hold-out**: Gradient Boosting, XGBoost, and Prophet.”
- “We report **MAE, MSE, RMSE, MAPE, and R²** for each.”
- “**Selection rule:** lowest RMSE, then MAPE. Only the winner drives the official 30-day forecast.”
- “On our run, **Gradient Boosting** typically wins; XGBoost is close; Prophet is weaker because this series is volatile and lag features matter.”
- “Residuals should concentrate around zero — we show the histogram so the jury can see error shape, not only a single score.”

**If asked “why not Isolation Forest in the table?”**
- “Isolation Forest detects outliers; it does not forecast revenue. Different task, different metrics — it appears in the anomaly section below.”

---

## 5) Anomaly detection — Student B (60 seconds)

**Do:**
- Stay on Forecasting; point to anomaly KPIs (counts / spikes / drops / rate)
- Point to the anomaly scatter plot

**Say:**
- “We flag abnormal days with **Isolation Forest** plus a **Z-score** safeguard.”
- “Spikes and drops give operations a shortlist of days to investigate — stock-outs, campaigns, logistics shocks.”
- “Example talking number: roughly **20** anomaly days across the calendar in our processed run.”

---

## 6) Segmentation — Student A (60–75 seconds)

**Do:**
- Open **Segmentation**
- Point to product scatter (revenue vs volatility) and customer RFM bar chart
- Optional: filter a segment or search a product id

**Say:**
- “Products are clustered with **K-Means** into high performers, seasonal opportunities, and at-risk SKUs.”
- “Customers use **RFM** logic so retention actions target dormant value, not random discounts.”
- “This is how BI segments feed the recommendation fact pack later.”

---

## 7) Recommendations (GRU + fact pack + Groq) — Student B (2 minutes) ⭐ critical

**Do:**
- Open **Recommendations**
- Point to GRU micro-F1 / sequence length metrics if shown
- Read **one** card fully (priority, theme, confidence, severity)
- Point to evidence line (“fact pack” / GRU score wording)

**Say:**
- “Recommendations are **not** a single hard-coded paragraph anymore.”
- “Step 1: a small **GRU** scores which themes matter from recent daily sequences.”
- “Step 2: we build a **fact pack** from forecast, anomalies, categories, customers, geography.”
- “Step 3: **Groq Llama 3.3 70B** writes the action and evidence text from those facts — or local NLG if the API is offline.”
- “Training used weak heuristic labels for supervision — we are transparent: inference is model-driven themes + data-driven wording.”
- “Confidence and severity help prioritise what management should act on first.”

**If Groq fails live:**
- “We designed an offline fallback so the academic demo never hard-depends on the network.”

---

## 8) Limitations (45–60 seconds) — both / Student B leads

**Say (pick 3–4):**
- “Data ends in **2018** — external validity is limited.”
- “Multi-step tree forecasts are recursive — errors can accumulate over 30 days.”
- “GRU labels are weakly supervised — fine for coursework, not a gold standard action ontology.”
- “LLM text is constrained by the fact pack; we still review outputs critically.”
- “Cluster names are interpretive business labels on K-Means groups.”

**Why say this:** markers reward critical thinking more than claiming perfection.

---

## 9) Closing (30 seconds)

**Say:**
- “Deliverables: reproducible pipeline, Streamlit lab, metrics artefacts, report, GitHub.”
- “We combine descriptive BI with evaluated AI — forecast selection, anomaly awareness, and actionable recommendations.”
- “Thank you — we’re happy to take questions.”

**Do:** return to Executive BI so the final visual is clean KPIs.

---

## Likely Q&A bullets (prepare aloud)

- **Train/test leak?** Cutoff `2018-05-01`; features use lags from history only; future forecast is recursive after full history fit for deployment path.
- **Why RMSE primary?** Penalises large business misses more than MAE alone.
- **Why MAPE secondary?** Interpretable % error for non-technical managers.
- **Confusion matrix?** Used for classification; our forecast is regression — residuals + MAE/RMSE/MAPE instead.
- **Is Groq “the model”?** No — Groq is the **wording** layer; GRU + analytics select themes and facts.
- **Who did what?** Point to the roles slide / report table without contradicting each other.

---

## Timing cheat sheet

| Block | Owner | Time |
| --- | --- | --- |
| Opening + problem | Both | ~1.5 min |
| Executive + Preparation | A | ~2.5 min |
| Forecast + residuals + anomalies | B | ~3.5 min |
| Segmentation | A | ~1 min |
| Recommendations | B | ~2 min |
| Limitations + close | Both | ~1.5 min |
