import pandas as pd
import numpy as np
from datetime import timedelta
# import holidays # Not strictly needed in this script if we only output date and RollCap

#Set random seed
np.random.seed(42)

# --- Configuration ---
HISTORICAL_DUMMY_DATA_PATH = 'dummy_data.csv'
OUTPUT_MOCK_ROLLCAP_PATH = 'mock_future_RollCap.csv' # Output file
NUM_FUTURE_MONTHS = 12 # Simulate for 12 months

print(f"--- Starting Generation of Mock Future RollCap Data for {NUM_FUTURE_MONTHS} months ---")

# --- 1. Load Base Historical Dummy Data ---
try:
    df_historical = pd.read_csv(HISTORICAL_DUMMY_DATA_PATH)
    df_historical['date'] = pd.to_datetime(df_historical['date'])
    print(f"Loaded historical dummy data: {df_historical.shape}")

    # Crucial check: 'capacity' column for RollCap calculation logic
    if 'capacity' not in df_historical.columns:
        print(f"ERROR: 'capacity' column not found in {HISTORICAL_DUMMY_DATA_PATH}. "
              "This script assumes 'RollCap' is a rolling sum of 'capacity'.")
        exit()
    if df_historical['capacity'].isnull().any():
        print(f"WARNING: 'capacity' column in {HISTORICAL_DUMMY_DATA_PATH} contains NaN values. Filling them.")
        df_historical['capacity'] = df_historical['capacity'].ffill().bfill()

    # Also check for existing 'RollCap' to get the last value, if calculation differs
    # For a consistent rolling sum, we just need last historical capacity values.
    if df_historical.empty:
        print("ERROR: Historical data is empty. Cannot simulate future values.")
        exit()

except FileNotFoundError:
    print(f"ERROR: Historical data file '{HISTORICAL_DUMMY_DATA_PATH}' not found. Please generate it first.")
    exit()
except Exception as e:
    print(f"An error occurred while loading historical data: {e}")
    exit()


# --- 2. Determine Future Date Range ---
last_historical_date = df_historical['date'].max()
future_start_date = last_historical_date + timedelta(days=1)

# Precise calculation for NUM_FUTURE_MONTHS to cover full months
if NUM_FUTURE_MONTHS <= 0:
    print("ERROR: NUM_FUTURE_MONTHS must be positive.")
    exit()
temp_end_date_month = future_start_date + pd.DateOffset(months=NUM_FUTURE_MONTHS - 1)
future_end_date = pd.Timestamp(year=temp_end_date_month.year, month=temp_end_date_month.month, day=1) + pd.offsets.MonthEnd(0)
future_dates = pd.date_range(start=future_start_date, end=future_end_date, freq='D')

if future_dates.empty:
    print("Error: Future date range is empty. Check NUM_FUTURE_MONTHS and historical data.")
    exit()
print(f"Simulating data for {len(future_dates)} days, from {future_dates.min().date()} to {future_dates.max().date()}")


# --- 3. Simulate Future Daily 'capacity' Values ---
# This logic should replicate how 'capacity' was generated in your Dummy_input.ipynb
print("\n--- Simulating Future Daily 'capacity' ---")
df_future_capacity = pd.DataFrame({'date': future_dates}) # Use a temporary df for future capacity
df_future_capacity['dayofyear'] = df_future_capacity['date'].dt.dayofyear
df_future_capacity['weekday_temp'] = df_future_capacity['date'].dt.dayofweek

# Parameters from your Dummy_input.ipynb for capacity generation
base_capacity = 25000
capacity_amplitude_year = 5000
capacity_amplitude_week = 2000 # This was defined in your snippet, ensure usage is intended
capacity_noise_std = 1000
weekday_factors_cap = np.array([1.05, 1.05, 1.0, 0.95, 0.9, 0.85, 0.9]) # Mon-Sun

capacity_yearly_trend = base_capacity + capacity_amplitude_year * np.sin(2 * np.pi * df_future_capacity['dayofyear'] / 365.25)
mean_weekday_factor_effect = base_capacity / weekday_factors_cap.mean()
capacity_weekly_component = weekday_factors_cap[df_future_capacity['weekday_temp'].values] * mean_weekday_factor_effect

df_future_capacity['capacity'] = (capacity_yearly_trend * 0.5 +
                                  capacity_weekly_component * 0.5 +
                                  np.random.normal(0, capacity_noise_std, size=len(df_future_capacity)))
df_future_capacity['capacity'] = np.maximum(20001, df_future_capacity['capacity']).astype(int)

print("Simulated future 'capacity' (first 5 rows):")
print(df_future_capacity[['date', 'capacity']].head())


# --- 4. Calculate Future 'RollCap' based on historical and simulated future 'capacity' ---
print("\n--- Calculating Future 'RollCap' ---")

# Get last 29 days of historical capacity (if available) to correctly start the rolling sum
if len(df_historical) >= 29 and 'capacity' in df_historical.columns:
    historical_capacity_for_rollcap = df_historical[['date', 'capacity']].iloc[-29:].copy()
elif 'capacity' in df_historical.columns: # Less than 29 days of history, but some
    historical_capacity_for_rollcap = df_historical[['date', 'capacity']].copy()
else: # No historical capacity, shouldn't happen based on earlier checks
    historical_capacity_for_rollcap = pd.DataFrame(columns=['date', 'capacity'])
    print("Warning: No historical 'capacity' data for rolling sum initialization. RollCap might be less accurate at the start.")

# Ensure 'date' is the index for proper concatenation and alignment for rolling calculation
historical_capacity_for_rollcap.set_index('date', inplace=True)
df_future_capacity_indexed = df_future_capacity.set_index('date')

# Combine historical (last 29 days) and future capacity
combined_capacity_series = pd.concat([
    historical_capacity_for_rollcap['capacity'],
    df_future_capacity_indexed['capacity']
])

# Calculate rolling sum on the combined series
rollcap_series_combined = combined_capacity_series.rolling(window=30, min_periods=1).sum()

# Extract the RollCap values that correspond to the future dates
simulated_future_rollcaps = rollcap_series_combined[df_future_capacity_indexed.index].values

# --- 5. Create and Save the Output DataFrame (Date and RollCap only) ---
df_output_mock_rollcap = pd.DataFrame({
    'date': future_dates, # These are the original future_dates we generated
    'RollCap': simulated_future_rollcaps
})

df_output_mock_rollcap['RollCap'] = df_output_mock_rollcap['RollCap'].bfill().astype(int) # Fill initial NaNs and ensure int

print("\nSimulated Future RollCap Data for Output (first 5 rows):")
print(df_output_mock_rollcap.head())
print("\nSimulated Future RollCap Data for Output (last 5 rows):")
print(df_output_mock_rollcap.tail())

try:
    df_output_mock_rollcap.to_csv(OUTPUT_MOCK_ROLLCAP_PATH, index=False, date_format='%Y-%m-%d')
    print(f"\nSuccessfully saved simulated future RollCap data (date, RollCap) to: {OUTPUT_MOCK_ROLLCAP_PATH}")
except Exception as e:
    print(f"\nError saving simulated future RollCap data: {e}")

print(f"\n--- Finished Generation of Mock Future RollCap Data ---")