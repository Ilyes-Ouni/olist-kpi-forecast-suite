from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bi_ai_retail.data_pipeline import run_pipeline


if __name__ == "__main__":
    outputs = run_pipeline()
    print("Pipeline completed successfully.")
    print(f"Processed files written to: {Path('data/processed').resolve()}")
    print(f"Forecasted next 30 days revenue: ${outputs['kpis']['forecast_30d_revenue']:,.2f}")
