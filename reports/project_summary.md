# Olist Commerce Decision Lab

## Executive Summary
- Total revenue: $11,461,734.20
- Total profit: $2,261,838.40
- Gross margin: 19.73%
- Orders analysed: 80,168
- Customers analysed: 77,775
- Repeat customer rate: 2.8%
- Top state: SP
- Top category: bed_bath_table
- Forecasted next 30 days revenue: $463,643.35
- Rows after cleaning: 95,351
- Noisy rows removed: 17,839

## Forecasting Performance
- MAE: 4926.73
- RMSE: 6046.08
- MAPE: 26.44%
- Forecast accuracy proxy: 80.46%

## GRU Recommendation Model
- Micro F1: 0.97
- Sequence length: 14
- Training windows: 477

## Management Actions
- [Low] Expansion: Use SP as the benchmark geography for assortment and service-level planning. (GRU confidence 100%. Largest geography revenue is $4,661,203.)
- [Medium] Customer Experience: Review product quality and delivery experience for security_and_services items. (GRU confidence 99%. Weakest category score is 2.50.)
- [Medium] Retention: Launch a targeted win-back campaign for dormant high-value customers. (GRU confidence 99%. About 18630 customers sit in the at-risk RFM segment.)
- [Medium] Portfolio: Rationalise low-performing SKUs through bundling, promotion, or discontinuation review. (GRU confidence 99%. Example at-risk product IDs: 46fce52cef5caa7cc225a5531c946c8b, 310dc32058903b6416c71faff132df9e, 0eeeb45e2f5911fd44282e5bb0c624ff.)