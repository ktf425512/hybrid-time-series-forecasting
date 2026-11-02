import os

import numpy as np
import pandas as pd

from _sim_config import (
    ROLLCAP_WINDOW,
    simulate_daily_capacity_and_total_presented,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_dummy_forecast_data(start_date_str, end_date_str, output_filepath, seed=42):
    """Generate synthetic daily call-volume data for forecasting demos."""
    print(f"Generating dummy data from {start_date_str} to {end_date_str}...")

    dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    capacity, total_presented = simulate_daily_capacity_and_total_presented(dates, seed=seed)

    rollcap = (
        pd.Series(capacity)
        .rolling(window=ROLLCAP_WINDOW, min_periods=1)
        .sum()
        .bfill()
        .astype(int)
        .values
    )

    df = pd.DataFrame({
        'date': dates,
        'Total_Presented': total_presented,
        'capacity': capacity,
        'RollCap': rollcap,
    })

    df.to_csv(output_filepath, index=False, date_format='%Y-%m-%d')
    print(f"Saved {len(df)} rows to {output_filepath}")
    print(df.head())


if __name__ == "__main__":
    generate_dummy_forecast_data(
        start_date_str='2022-01-01',
        end_date_str='2024-12-31',
        output_filepath=os.path.join(BASE_DIR, 'dummy_data.csv'),
    )
