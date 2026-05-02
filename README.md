# Olist Commerce Decision Lab

Olist Commerce Decision Lab is a BI + AI project that transforms retail e-commerce data into an interactive decision-support system. It combines business intelligence reporting with machine learning to monitor performance, forecast sales, detect anomalies, segment customers and products, and generate actionable recommendations.

## Repository

**GitHub:** [Ilyes-Ouni/olist-kpi-forecast-suite](https://github.com/Ilyes-Ouni/olist-kpi-forecast-suite)

Canonical clone URL:

```text
https://github.com/Ilyes-Ouni/olist-kpi-forecast-suite.git
```

### Recent updates (branding & docs)

- Renamed public project identity from **Retail Decision Intelligence Platform** to **Olist Commerce Decision Lab** (distinct naming; Olist‑specific focus).
- Refreshed **Streamlit** app title and `page_title` in `app.py`.
- Aligned **`README.md`**, **`reports/`**, **`presentation/`** outline copy, **`src/bi_ai_retail/reporting.py`** Markdown reports, and **`presentation/generate_presentation.py`** title slide/caption metadata.
- Presentation export filename set to **`Olist_Commerce_Decision_Lab_Presentation.pptx`**; subtitle text uses the Brazilian e‑commerce (**Olist**) dataset wording.

## Project Overview

This project was developed as an academic BI + AI solution using the Olist e-commerce dataset. The objective is to combine descriptive analytics and predictive analytics in one platform so that business performance can be monitored and future-oriented decisions can be supported with data.

The platform helps answer questions such as:

- Which product categories and states generate the most revenue?
- How do delivery performance and customer reviews affect business quality?
- What sales level can be expected in the next 30 days?
- Which unusual sales patterns should be investigated?
- Which customers and products require strategic attention?

## Objectives

- Prepare and clean e-commerce transaction data for analysis
- Build business KPIs for revenue, orders, reviews, and delivery performance
- Forecast future sales using machine learning
- Detect anomalies in sales behavior
- Segment customers and products for decision support
- Demonstrate advanced preprocessing techniques such as scaling, feature fusion, noise removal, and PCA
- Deliver an interactive dashboard for academic presentation

## Dataset

- Source: [Olist merged e-commerce dataset](https://huggingface.co/datasets/abhimlv/Olist-preprocessed-data-merged/resolve/main/orders_final_merged_prepro_and_feature_engg.csv)
- Type: Brazilian e-commerce transactional dataset
- Period covered: `2016-10-03` to `2018-08-29`

The dataset includes order, payment, review, customer, seller, freight, and product-related features.

## Methodology

### 1. Data Preparation

The preprocessing stage includes:

- type conversion for date, numeric, and categorical fields
- missing-value treatment with median and default-category imputation
- duplicate detection and removal
- IQR-based noise filtering on key numerical variables
- feature fusion and derived variables
- MinMax scaling
- Standard scaling
- PCA-based dimension reduction

### 2. Feature Engineering

Examples of engineered features include:

- `product_volume_cm3`
- `product_density_g_cm3`
- `freight_per_weight`
- `payment_per_installment`
- `review_text_length`
- `approval_lag_hours`
- `delivery_delay_days`
- `price_to_freight_ratio`

### 3. Business Intelligence Layer

The BI component provides:

- revenue and order KPIs
- average review and late-delivery indicators
- state and regional performance analysis
- category performance analysis
- trend monitoring over time

### 4. Artificial Intelligence Layer

The AI component includes:

- daily sales forecasting using machine learning
- anomaly detection for unusual revenue patterns
- customer segmentation using RFM logic
- product segmentation using clustering
- recommendation logic based on analytical outputs

### 5. Dashboard

The dashboard was developed with Streamlit and includes:

- executive BI overview
- data preparation and preprocessing analysis
- forecasting and anomaly detection views
- customer and product segmentation views
- recommendation panel

## Tools and Technologies

- Python
- Pandas
- NumPy
- scikit-learn
- Plotly
- Streamlit
- Joblib
- python-pptx

## Project Structure

```text
olist-kpi-forecast-suite/
|-- app.py
|-- main.py
|-- requirements.txt
|-- dashboard/
|-- notebooks/
|-- presentation/
|-- reports/
`-- src/bi_ai_retail/
```

## How to Run

Install dependencies:

```powershell
py -3.14 -m pip install -r requirements.txt
```

Run the data pipeline:

```powershell
py -3.14 main.py
```

Launch the dashboard:

```powershell
py -3.14 -m streamlit run app.py
```

## Main Outputs

The project generates:

- cleaned analytical dataset
- KPI summaries
- forecast results
- anomaly outputs
- customer and product segments
- preprocessing outputs for scaling and PCA
- business recommendations

Additional preprocessing files include:

- `missing_value_report.csv`
- `scaler_summary.csv`
- `minmax_scaled_features.csv`
- `standard_scaled_features.csv`
- `pca_projection.csv`
- `preprocessing_summary.json`

## Results Summary

- Revenue analysed: `$11.46M`
- Profit analysed: `$2.26M`
- Gross margin: `19.7%`
- Orders analysed: `80,168`
- Customers analysed: `77,775`
- States analysed: `27`
- Repeat customer rate: `2.8%`
- Forecasted revenue for the next 30 days: `$463,643`
- Detected anomaly days: `20`
- Rows after cleaning: `95,351`
- Noisy rows removed: `17,839`

## Recommendation Logic

The recommendation module is rule-based. It uses results from forecasting, anomaly detection, category analysis, customer segmentation, product segmentation, and geography performance to generate business actions such as:

- inventory adjustment
- operations investigation
- customer experience improvement
- product portfolio review
- customer retention actions
- geography benchmarking

## Academic Value

This project demonstrates practical skills in:

- data cleaning and preprocessing
- feature engineering
- KPI design
- business intelligence reporting
- machine learning for forecasting
- anomaly detection
- customer and product segmentation
- dashboard development
- business-oriented data storytelling

## Conclusion

Olist Commerce Decision Lab shows how BI and AI can be combined in a realistic e-commerce use case. The BI component explains what has happened in the business, while the AI component provides predictive and analytical support for decision-making.
