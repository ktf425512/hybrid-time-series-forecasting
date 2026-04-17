"""Generate mock future RollCap (30-day rolling sum of capacity)."""
import os
import sys
from datetime import timedelta

import pandas as pd

from _sim_config import DEFAULT_SEED, ROLLCAP_WINDOW, simulate_daily_capacity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_DUMMY_DATA_PATH = os.path.join(BASE_DIR, 'dummy_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'mock_future_RollCap.csv')
NUM_FUTURE_MONTHS = 12


def main():
    print(f"--- Generating {NUM_FUTURE_MONTHS} months of mock future RollCap ---")

    try:
        df_historical = pd.read_csv(HISTORICAL_DUMMY_DATA_PATH, parse_dates=['date'])
    except FileNotFoundError:
        print(f"ERROR: '{HISTORICAL_DUMMY_DATA_PATH}' not found. Generate it first.")
        sys.exit(1)

    if 'capacity' not in df_historical.columns or df_historical.empty:
        print("ERROR: historical data missing 'capacity' column or is empty.")
        sys.exit(1)

    last_hist_date = df_historical['date'].max()
    future_start = last_hist_date + timedelta(days=1)
    future_end = (future_start + pd.DateOffset(months=NUM_FUTURE_MONTHS - 1)).replace(day=1) + pd.offsets.MonthEnd(0)
    future_dates = pd.date_range(start=future_start, end=future_end, freq='D')

    if future_dates.empty:
        print("ERROR: future date range is empty.")
        sys.exit(1)

    print(f"Simulating {len(future_dates)} days from {future_dates.min().date()} to {future_dates.max().date()}")
    future_capacity = simulate_daily_capacity(future_dates, seed=DEFAULT_SEED)

    # Seed the rolling window with the tail of historical capacity so the first
    # future RollCap values don't start from a partial window.
    hist_tail = df_historical[['date', 'capacity']].iloc[-(ROLLCAP_WINDOW - 1):].set_index('date')['capacity']
    future_capacity_series = pd.Series(future_capacity, index=future_dates)
    combined = pd.concat([hist_tail, future_capacity_series])

    rollcap_all = combined.rolling(window=ROLLCAP_WINDOW, min_periods=1).sum()
    future_rollcap = rollcap_all.loc[future_dates].bfill().astype(int).values

    out = pd.DataFrame({'date': future_dates, 'RollCap': future_rollcap})
    out.to_csv(OUTPUT_PATH, index=False, date_format='%Y-%m-%d')
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    print(out.head())


if __name__ == "__main__":
    main()
