import pandas as pd
import numpy as np
from datetime import timedelta
# from dateutil.relativedelta import relativedelta # pd.DateOffset is generally sufficient

# Set the random seed for reproducibility of noise component
np.random.seed(42)

# --- Configuration ---
# Path to your base historical dummy data.
# This is used to determine the start date for future simulation.
# If not found, a fallback (current date) is used.
HISTORICAL_DUMMY_DATA_PATH = 'dummy_direct_data.csv'
OUTPUT_MOCK_MONTHLY_SUM_PATH = 'mock_future_Month_Sum.csv' # Output file
NUM_FUTURE_MONTHS = 12 # Simulate for 12 future months

# Parameters for generating daily 'Total_Presented'
# These should be IDENTICAL to how your dummy_direct_data.csv's Total_Presented was generated.
BASE_CALLS = 90000
CALLS_AMPLITUDE_YEAR = 10000
# CALLS_AMPLITUDE_WEEK = 5000 # This parameter was defined in your original snippet but not explicitly
                             # used in the 'calls_weekly_trend' formula you provided.
                             # If it's intended to modulate the weekly effect,
                             # the 'calls_weekly_component' formula would need adjustment.
CALLS_NOISE_STD = 2000
YEARLY_TREND_PHASE_SHIFT = np.pi / 4 # Phase shift for yearly sine wave (e.g., to shift peak from Jan 1)
WEEKDAY_FACTORS_CALLS = np.array([1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.85]) # Mon-Sun factors
MIN_TOTAL_PRESENTED = 80001 # Minimum allowed value for Total_Presented

print(f"--- Starting Generation of Mock Future Monthly Sum Data for {NUM_FUTURE_MONTHS} months ---")
print(f"Using 'bottom-up' approach: simulating daily Total_Presented then aggregating.")

# --- 1. Determine Future Date Range (Daily level for NUM_FUTURE_MONTHS) ---
try:
    # Try to load historical data just to get the last date as a starting point for the future.
    df_historical_for_date = pd.read_csv(HISTORICAL_DUMMY_DATA_PATH)
    last_historical_date = pd.to_datetime(df_historical_for_date['date']).max()
    print(f"Last date from historical data: {last_historical_date.date()}")
except FileNotFoundError:
    print(f"Warning: '{HISTORICAL_DUMMY_DATA_PATH}' not found. Using current date ({pd.Timestamp.now().normalize().date()}) as reference for future start.")
    last_historical_date = pd.Timestamp.now().normalize() # Use today's date at midnight as fallback
except Exception as e:
    print(f"Warning: Error loading historical data to get last date: {e}. Using current date ({pd.Timestamp.now().normalize().date()}) as reference.")
    last_historical_date = pd.Timestamp.now().normalize()


future_start_date_daily = last_historical_date + timedelta(days=1)

if NUM_FUTURE_MONTHS <= 0:
    print("ERROR: NUM_FUTURE_MONTHS must be positive.")
    exit()

# Calculate the end date to precisely cover NUM_FUTURE_MONTHS
# The daily range should go up to the last day of the NUM_FUTURE_MONTHS-th month.
temp_end_month_for_daily_range = future_start_date_daily + pd.DateOffset(months=NUM_FUTURE_MONTHS - 1)
future_end_date_daily = pd.Timestamp(year=temp_end_month_for_daily_range.year,
                                     month=temp_end_month_for_daily_range.month,
                                     day=1) + pd.offsets.MonthEnd(0) # Get the last day of that month

future_daily_dates = pd.date_range(start=future_start_date_daily, end=future_end_date_daily, freq='D')

if future_daily_dates.empty:
    print("Error: Future daily date range is empty. Check NUM_FUTURE_MONTHS and calculated start/end dates.")
    exit()

print(f"Generating daily Total_Presented for {len(future_daily_dates)} days, from {future_daily_dates.min().date()} to {future_daily_dates.max().date()}")

# --- 2. Simulate Future Daily 'Total_Presented' Values ---
# This logic should precisely mirror how 'Total_Presented' was generated in your Dummy_input.ipynb
df_future_daily_tp = pd.DataFrame({'date': future_daily_dates})
df_future_daily_tp['dayofyear'] = df_future_daily_tp['date'].dt.dayofyear
df_future_daily_tp['weekday_temp'] = df_future_daily_tp['date'].dt.dayofweek # Monday=0, Sunday=6

# Yearly trend component for Total_Presented
calls_yearly_trend = BASE_CALLS + CALLS_AMPLITUDE_YEAR * np.sin(
    2 * np.pi * df_future_daily_tp['dayofyear'] / 365.25 + YEARLY_TREND_PHASE_SHIFT
)

# Weekly trend component for Total_Presented
# Ensure WEEKDAY_FACTORS_CALLS has length 7
if len(WEEKDAY_FACTORS_CALLS) != 7:
    print("ERROR: WEEKDAY_FACTORS_CALLS must have 7 elements (Mon-Sun).")
    exit()
# This scaling ensures that applying factors to BASE_CALLS, on average, results in BASE_CALLS.
mean_weekday_factor_effect = BASE_CALLS / WEEKDAY_FACTORS_CALLS.mean()
calls_weekly_component = WEEKDAY_FACTORS_CALLS[df_future_daily_tp['weekday_temp'].values] * mean_weekday_factor_effect
# Note on CALLS_AMPLITUDE_WEEK: If this was intended to make the weekly swing more pronounced or dampened,
# the formula for calls_weekly_component would be:
# weekly_swing = (WEEKDAY_FACTORS_CALLS[df_future_daily_tp['weekday_temp'].values] - WEEKDAY_FACTORS_CALLS.mean()) * CALLS_AMPLITUDE_WEEK
# calls_weekly_component = BASE_CALLS + weekly_swing
# For now, using the formula structure derived from your provided snippet.

# Combine components for 'Total_Presented'
df_future_daily_tp['Total_Presented'] = (calls_yearly_trend * 0.4 +  # Weighting from your formula
                                         calls_weekly_component * 0.6 + # Weighting from your formula
                                         np.random.normal(0, CALLS_NOISE_STD, size=len(df_future_daily_tp)))
df_future_daily_tp['Total_Presented'] = np.maximum(MIN_TOTAL_PRESENTED, df_future_daily_tp['Total_Presented']).astype(int)

print("\nSimulated future daily 'Total_Presented' (first 5 rows):")
print(df_future_daily_tp[['date', 'Total_Presented']].head())

# --- 3. Aggregate Simulated Daily 'Total_Presented' to Monthly Sums ---
print("\n--- Aggregating simulated daily Total_Presented to monthly sums ---")
# Set 'date' as index for resampling
df_future_daily_tp_indexed = df_future_daily_tp.set_index('date')
# Resample to get sum for each month; the index of the result will be the start of the month (MS)
simulated_monthly_sums_series = df_future_daily_tp_indexed['Total_Presented'].resample('MS').sum()

print("\nSimulated future monthly sums (all generated months):")
print(simulated_monthly_sums_series)

# --- 4. Create and Save the Output DataFrame ---
# The output format should be 'date' (start of month), 'monthly_sum', and 'year_month'
df_mock_future_monthly_sum = pd.DataFrame({
    'date': simulated_monthly_sums_series.index, # Index is already month-start Timestamps
    'monthly_sum': simulated_monthly_sums_series.values
})

# Create the 'year_month' column in 'YYYY-MM' format
df_mock_future_monthly_sum['year_month'] = pd.to_datetime(df_mock_future_monthly_sum['date']).dt.to_period('M').astype(str)

df_mock_future_monthly_sum['monthly_sum'] = df_mock_future_monthly_sum['monthly_sum'].astype(int)

# Reorder columns to match your target format (as per image_4980fc.png)
df_mock_future_monthly_sum = df_mock_future_monthly_sum[['date', 'monthly_sum', 'year_month']]

print("\nFinal Simulated Future Monthly Sum Data for Output (first 5 rows):")
print(df_mock_future_monthly_sum.head())
if len(df_mock_future_monthly_sum) > 5: # Show tail if more than 5 months
    print("\nFinal Simulated Future Monthly Sum Data for Output (last 5 rows):")
    print(df_mock_future_monthly_sum.tail())

try:
    df_mock_future_monthly_sum.to_csv(OUTPUT_MOCK_MONTHLY_SUM_PATH, index=False, date_format='%Y-%m-%d')
    print(f"\nSuccessfully saved simulated future monthly sum data to: {OUTPUT_MOCK_MONTHLY_SUM_PATH}")
except Exception as e:
    print(f"\nError saving simulated future monthly sum data: {e}")

print(f"\n--- Finished Generation of Mock Future Monthly Sum Data ---")