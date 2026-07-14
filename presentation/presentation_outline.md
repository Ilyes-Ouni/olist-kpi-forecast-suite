# Presentation Outline

## Slide 1: Title

Olist Commerce Decision Lab  
BI + AI Academic Project (2-student team)

## Slide 2: Context and Problem

- Managers often see only historical dashboards.
- Past KPIs do not answer “what next?” or “what is abnormal?”.
- Need one platform for monitoring + prediction + action.

## Slide 3: Proposed Solution

- BI KPIs, geography, categories, trends
- Multi-model demand forecasting (GBR / XGBoost / Prophet) + selection rule
- Anomaly detection (Isolation Forest + Z-score)
- Customer RFM + product K-Means segmentation
- Hybrid recommendations: GRU → fact pack → Groq / local NLG
- Interactive Streamlit demo

## Slide 4: Dataset

- Brazilian e-commerce (Olist) merged transaction data
- Orders, payments, reviews, customers, sellers, freight, products
- Period: 2016-10-03 → 2018-08-29

## Slide 5: Technical Workflow

1. Cleaning & preprocessing (IQR, feature fusion, scaling, PCA)
2. BI KPI & analytical tables
3. Forecast candidates + hold-out evaluation
4. Anomaly detection
5. Segmentation
6. GRU + fact-pack recommendations
7. Streamlit dashboard

## Slide 6: Team roles

- **Student A (BI):** preparation, KPIs, dashboard storytelling, filters
- **Student B (AI):** forecast comparison, anomalies, GRU/Groq recommendations, metrics defence

## Slide 7: BI Layer

- Revenue, profit, orders, customers, late delivery, reviews
- State / category performance
- Time trends with filters

## Slide 8: Forecasting evaluation

- Same train/test cutoff for all models
- Metrics: MAE, MSE, RMSE, MAPE, R²
- Rule: **lowest RMSE, then MAPE**
- Residual histogram for the winner
- Isolation Forest **not** listed here (wrong task type)

## Slide 9: Why the winner (typically Gradient Boosting)

- Lag / rolling features capture short-term demand better than smooth seasonality alone
- Prophet can underfit volatile daily revenue
- XGBoost is close; GBR often edges RMSE on this hold-out

## Slide 10: Anomaly detection

- Isolation Forest + Z-score on daily signals
- Spike vs Drop labels
- Show counts + % of days (support ops investigation)

## Slide 11: Recommendations pipeline

- GRU scores themes from recent sequences
- Fact pack injects live evidence
- Groq (or local NLG) writes action + evidence text
- Confidence / severity shown in UI

## Slide 12: Limitations

- Data ends 2018
- Recursive multi-step forecast error growth
- Weak GRU training labels
- LLM needs fact constraints + offline fallback
- Cluster labels are interpretive

## Slide 13: Deliverables

- Pipeline + dashboard + processed outputs
- Model comparison CSV / metrics JSON
- Report + demo script + GitHub

## Slide 14: Conclusion

- BI explains the past; AI supports the next decision
- Explicit comparison + limitations = academically defensible
- Demo-ready decision lab for Olist retail analytics

## Live demo checklist

Use `presentation/demo_script.md` for the exact talk track and click path.
