import pandas as pd
import numpy as np
import datetime
import os # Added os import for potential path handling if needed later

def generate_dummy_forecast_data(start_date_str, end_date_str, output_filepath):
    """
    Generates dummy time series data for call volume forecasting demo.

    Args:
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.
        output_filepath (str): Path to save the generated CSV file.
    """
    print(f"Generating dummy data from {start_date_str} to {end_date_str}...")

    # --- Date Range ---
    try:
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
        df = pd.DataFrame({'date': dates})
    except ValueError as e:
        print(f"Error creating date range: {e}")
        return

    # --- Generate Core Business Metrics ---

    # Seed for reproducibility
    np.random.seed(42)

    # Temporary date features for seasonality calculation
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekday_temp'] = df['date'].dt.dayofweek # Monday=0, Sunday=6

    # 1. Generate 'capacity' (ensure > 20k)
    base_capacity = 25000
    capacity_amplitude_year = 5000
    capacity_amplitude_week = 2000
    capacity_noise_std = 1000
    capacity_yearly_trend = base_capacity + capacity_amplitude_year * np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    weekday_factors_cap = np.array([1.05, 1.05, 1.0, 0.95, 0.9, 0.85, 0.9]) # Mon-Sun factors
    capacity_weekly_trend = weekday_factors_cap[df['weekday_temp']] * (base_capacity / (weekday_factors_cap.mean()))
    df['capacity'] = (capacity_yearly_trend * 0.5 + capacity_weekly_trend * 0.5 +
                      np.random.normal(0, capacity_noise_std, size=len(df)))
    df['capacity'] = np.maximum(20001, df['capacity']).astype(int)

    # 2. Calculate 'Rollcap' (Simple trailing 30-day sum of capacity)
    df['RollCap'] = df['capacity'].rolling(window=30, min_periods=1).sum()
    df['RollCap'] = df['RollCap'].bfill().astype(int) # Use recommended .bfill()

    # 3. Generate 'Total_Presented' (ensure > 80k)
    base_calls = 90000
    calls_amplitude_year = 10000
    calls_amplitude_week = 5000
    calls_noise_std = 2000
    calls_yearly_trend = base_calls + calls_amplitude_year * np.sin(2 * np.pi * df['dayofyear'] / 365.25 + np.pi/4) # Phase shift
    weekday_factors_calls = np.array([1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.85]) # Mon-Sun factors
    calls_weekly_trend = weekday_factors_calls[df['weekday_temp']] * (base_calls / (weekday_factors_calls.mean()))
    df['Total_Presented'] = (calls_yearly_trend * 0.4 + calls_weekly_trend * 0.6 +
                             np.random.normal(0, calls_noise_std, size=len(df)))
    df['Total_Presented'] = np.maximum(80001, df['Total_Presented']).astype(int)

    # --- Final Column Selection (ONLY the 4 core + date) ---
    final_columns = ['date', 'Total_Presented', 'capacity', 'RollCap']
    # Drop temporary columns before selecting final ones
    df_final = df[final_columns].copy()

    # --- Save to CSV ---
    try:
        # Ensure directory exists if filepath includes directories
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save with standard date format YYYY-MM-DD
        df_final.to_csv(output_filepath, index=False, date_format='%Y-%m-%d')
        print(f"\nDummy data generated successfully with {len(df_final)} rows.")
        print(f"Columns: {df_final.columns.tolist()}")
        print(f"Saved to {output_filepath}")
        print("\nFirst 5 rows of generated data:")
        print(df_final.head())
    except Exception as e:
        print(f"\nError saving file to {output_filepath}: {e}")
        print("Please check the file path and write permissions.")

# --- Main execution block ---
if __name__ == "__main__":
    # Configuration
    start_date_str = '2022-01-01'
    end_date_str = '2024-12-31'
    # Define where to save the output file relative to the script location
    # Saves in the same directory as the script by default
    output_filename = 'dummy_data.csv'

    # Generate the data
    generate_dummy_forecast_data(start_date_str, end_date_str, output_filename)