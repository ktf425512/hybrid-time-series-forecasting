"""Generate mock future monthly sums of Total_Presented.

Bottom-up approach: simulate daily Total_Presented for the future horizon
using the same formula as the historical data, then aggregate to months.
"""
import os
import sys
from datetime import timedelta

import pandas as pd

from _sim_config import DEFAULT_SEED, simulate_daily_total_presented

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_DUMMY_DATA_PATH = os.path.join(BASE_DIR, 'dummy_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'future_Month_Sum.csv')
NUM_FUTURE_MONTHS = 12


def main():
    print(f"--- Generating {NUM_FUTURE_MONTHS} months of mock future monthly sums ---")

    try:
        last_hist_date = pd.to_datetime(pd.read_csv(HISTORICAL_DUMMY_DATA_PATH)['date']).max()
    except FileNotFoundError:
        print(f"'{HISTORICAL_DUMMY_DATA_PATH}' not found. Using today as reference.")
        last_hist_date = pd.Timestamp.now().normalize()

    future_start = last_hist_date + timedelta(days=1)
    future_end = (future_start + pd.DateOffset(months=NUM_FUTURE_MONTHS - 1)).replace(day=1) + pd.offsets.MonthEnd(0)
    future_dates = pd.date_range(start=future_start, end=future_end, freq='D')

    if future_dates.empty:
        print("ERROR: future date range is empty.")
        sys.exit(1)

    print(f"Simulating {len(future_dates)} days from {future_dates.min().date()} to {future_dates.max().date()}")
    total_presented = simulate_daily_total_presented(future_dates, seed=DEFAULT_SEED)

    daily = pd.DataFrame({'date': future_dates, 'Total_Presented': total_presented})
    monthly_sums = daily.set_index('date')['Total_Presented'].resample('MS').sum()

    out = pd.DataFrame({
        'date': monthly_sums.index,
        'monthly_sum': monthly_sums.values.astype(int),
        'year_month': monthly_sums.index.to_period('M').astype(str),
    })
    out.to_csv(OUTPUT_PATH, index=False, date_format='%Y-%m-%d')
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    print(out)


if __name__ == "__main__":
    main()
