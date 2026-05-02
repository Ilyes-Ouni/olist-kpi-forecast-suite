# Final Report

## Project title

Olist Commerce Decision Lab

## Problem statement

Retail organisations usually monitor past sales through descriptive dashboards, but they still struggle to anticipate future demand and quickly react to abnormal sales patterns. This project addresses that gap by combining Business Intelligence and Artificial Intelligence in one decision-support platform.

## Objectives

1. Prepare a clean retail analytics dataset from an open source transaction file.
2. Build KPI tables and business views for revenue, profit, and geographic performance.
3. Forecast short-term demand using machine learning.
4. Detect unusual sales spikes and drops.
5. Segment products into strategic groups.
6. Generate recommendations for business actions.
7. Deliver an interactive dashboard for decision-makers.

## Methodology

### Data preparation

- Removed duplicates
- Filtered cancelled or invalid transactions
- Parsed invoice dates
- Created revenue, estimated cost, profit, and margin fields
- Derived region and business category dimensions

### BI analytics

- Executive KPIs
- Country and region performance
- Category performance
- Monthly trend analysis

### AI analytics

- Forecasting with Gradient Boosting Regressor
- Anomaly detection with Isolation Forest and Z-score
- Product clustering with K-Means
- Rule-based business recommendations

## Deliverables

- Runnable Python pipeline
- Interactive Streamlit dashboard
- Processed CSV outputs
- Project summary report
- Presentation outline

## Conclusion

The project demonstrates how BI and AI complement each other in a realistic retail setting. BI explains performance and supports reporting, while AI adds prediction, pattern discovery, and proactive decision support.

