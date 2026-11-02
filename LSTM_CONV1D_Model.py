"""Conv1D + LSTM daily call-volume forecasting pipeline.

Train a daily call-volume model with a manual TimeSeriesSplit CV over a
hyperparameter grid and generate a 12-month forward forecast via iterative
1-step-ahead prediction.

The project framing is *disaggregation*: at inference we have an external
monthly-total forecast (``future_Month_Sum.csv``) and the model learns the
daily breakdown within it. See ``inject_monthly_sum_noise`` for the reason
we perturb the training-time monthly_sum signal.
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')  # write PNGs, never display
import holidays
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')        # where generate_*.py write
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')   # created on first run

# Inputs (relative to --data-dir)
HISTORICAL_DATA_FILENAME = 'dummy_data.csv'
FUTURE_ROLLCAP_FILENAME = 'mock_future_RollCap.csv'
FUTURE_MONTH_SUM_FILENAME = 'future_Month_Sum.csv'

# Outputs (relative to --output-dir)
SCALER_CV_FILENAME = 'scaler_cv.pkl'
SCALER_FINAL_FILENAME = 'scaler_final.pkl'
MODEL_FINAL_FILENAME = 'model_final.keras'
CV_TEST_PLOT_FILENAME = 'cv_test_actual_vs_predicted.png'
FORECAST_PLOT_FILENAME = 'forecast_vs_historical.png'
FORECAST_CSV_FILENAME = 'forecast.csv'

# --- Holiday settings ---
HOLIDAY_YEARS = list(range(2021, 2028))
US_HOLIDAYS_SUBSET = [
    "New Year's Day", "New Year's Day (observed)", "Memorial Day",
    "Independence Day", "Labor Day", "Thanksgiving", "Christmas Day",
]
HOLIDAY_A_NAMES = ["Thanksgiving", "Christmas Day"]

# --- Features (single source of truth; scaled subset derived by name) ---
FEATURE_ORDER = (
    'Total_Presented',   # target (index 0)
    'RollCap',
    'Holiday_A',
    'Holiday_B',
    'weekday',
    'week_number',
    'month',
    'monthly_sum',
)
_SCALED_FEATURES = {'Total_Presented', 'RollCap', 'monthly_sum'}
COLUMNS_TO_SCALE = [c for c in FEATURE_ORDER if c in _SCALED_FEATURES]
TARGET_COLUMN_NAME = FEATURE_ORDER[0]
N_FEATURES = len(FEATURE_ORDER)

# --- Windowing / training ---
N_PAST_STEPS = 7
TRAIN_SPLIT_RATIO_CV = 0.95
CV_N_SPLITS = 3
# Hyperparameter grid — kept to a single combo on this stack because repeated
# Keras model.fit() calls in the same process trigger a native-layer crash
# on Windows (Python exits 127/139 after ~2-3 fits, no Python traceback, due
# to TF graph accumulation that clear_session doesn't fully release). The
# manual CV loop still runs CV_N_SPLITS folds of this one combo as a TSS
# sanity check. To explore more hyperparameters, run the pipeline repeatedly
# with different values (each run is a fresh process) or port the search to
# a subprocess-per-fit worker.
GRID_SEARCH_PARAMS = {
    'batch_size': [16],
    'epochs': [30],
    'optimizer': ['adam'],
    'learning_rate': [0.001],
    'l2_coeff': [0.01],
}
EARLY_STOPPING_PATIENCE = 10
FIXED_L2_COEFF = 0.01

# Disaggregation framing: at inference, monthly_sum comes from an external
# forecast that typically drifts 5-10%. Training on the exact sum-of-target
# (noise-free) teaches the model to over-rely on a signal it won't have in
# production. We inject per-month multiplicative jitter to close the gap.
MONTHLY_SUM_NOISE_STD = 0.05
MONTHLY_SUM_NOISE_SEED = 42

PIPELINE_SEED = 42


def load_holidays_data(years, holiday_a_names_list, us_holidays_subset_list):
    """Build per-date Holiday_A/Holiday_B flags from the `holidays` package."""
    items = list(holidays.UnitedStates(years=years).items())
    df_h = pd.DataFrame(items, columns=['date', 'holiday_name'])
    df_h['date'] = pd.to_datetime(df_h['date'])
    df_h.drop_duplicates(subset=['date'], keep='first', inplace=True)

    df_h['Holiday_A'] = df_h['holiday_name'].isin(holiday_a_names_list).astype(int)
    df_h['Holiday_B'] = (
        df_h['holiday_name'].isin(us_holidays_subset_list)
        & ~df_h['holiday_name'].isin(holiday_a_names_list)
    ).astype(int)
    return df_h[['date', 'Holiday_A', 'Holiday_B']]


def engineer_features_historical(filepath, df_holidays):
    """Load historical CSV and join calendar + monthly_sum features."""
    df = pd.read_csv(filepath, parse_dates=['date'])

    df = pd.merge(df, df_holidays, on='date', how='left')
    df['Holiday_A'] = df['Holiday_A'].fillna(0).astype(int)
    df['Holiday_B'] = df['Holiday_B'].fillna(0).astype(int)

    df['date_dt'] = pd.to_datetime(df['date'])
    df['month_start_key'] = df['date_dt'].dt.to_period('M').dt.to_timestamp()
    df['weekday'] = df['date_dt'].dt.dayofweek
    df['week_number'] = df['date_dt'].dt.isocalendar().week.astype(int)
    df['month'] = df['date_dt'].dt.month

    monthly_sum_series = (
        df.set_index('date_dt')[TARGET_COLUMN_NAME]
        .resample('MS').sum()
        .rename('monthly_sum_val')
    )
    df = pd.merge(df, monthly_sum_series, left_on='month_start_key', right_index=True, how='left')
    df = df.rename(columns={'monthly_sum_val': 'monthly_sum'})
    df['monthly_sum'] = df['monthly_sum'].ffill().bfill().fillna(0)

    df = df.set_index('date_dt')
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    return df[list(FEATURE_ORDER)]


def inject_monthly_sum_noise(df, noise_std=MONTHLY_SUM_NOISE_STD, seed=MONTHLY_SUM_NOISE_SEED):
    """Multiply each calendar month's monthly_sum by one shared jitter factor.

    One factor per month (not per day) preserves the "monthly total" semantics.
    Simulates the noisy external monthly forecast seen in production.
    """
    months = df.index.to_period('M')
    unique_months = list(months.unique())
    rng = np.random.default_rng(seed)
    factors = rng.normal(1.0, noise_std, size=len(unique_months))
    factor_map = dict(zip(unique_months, factors))
    factor_series = pd.Series([factor_map[m] for m in months], index=df.index)

    df = df.copy()
    df['monthly_sum'] = df['monthly_sum'] * factor_series
    return df


def create_sequences_from_data(dataset_values, n_past, n_features_expected, target_col_idx):
    """Emit windowed (X, y): X has shape (n_past, n_features); y is the next target."""
    if len(dataset_values) <= n_past:
        print(f"Warning: dataset length {len(dataset_values)} too short for n_past={n_past}")
        return np.empty((0, n_past, n_features_expected)), np.empty((0,))

    if dataset_values.shape[1] != n_features_expected:
        raise ValueError(
            f"Dataset has {dataset_values.shape[1]} features, expected {n_features_expected}"
        )

    data_x, data_y = [], []
    for i in range(n_past, len(dataset_values)):
        data_x.append(dataset_values[i - n_past:i, :])
        data_y.append(dataset_values[i, target_col_idx])
    return np.array(data_x), np.array(data_y)


def scale_data_and_create_sequences(df_full, train_ratio, n_past, cols_to_scale_list,
                                    ordered_features_list, target_col_name):
    """Split, scale (fit on train only), and emit windowed (X, y) arrays."""
    split_idx = int(len(df_full) * train_ratio)
    df_train = df_full.iloc[:split_idx]
    df_test = df_full.iloc[split_idx:]

    print(f"Training set size: {len(df_train)}, Testing set size: {len(df_test)}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[cols_to_scale_list] = scaler.fit_transform(df_train[cols_to_scale_list])
    if not df_test.empty:
        df_test_scaled[cols_to_scale_list] = scaler.transform(df_test[cols_to_scale_list])

    train_values = df_train_scaled[list(ordered_features_list)].values
    test_values = df_test_scaled[list(ordered_features_list)].values

    n_features = len(ordered_features_list)
    target_col_idx = ordered_features_list.index(target_col_name)

    x_train, y_train = create_sequences_from_data(train_values, n_past, n_features, target_col_idx)
    x_test, y_test = create_sequences_from_data(test_values, n_past, n_features, target_col_idx)

    return (
        x_train.astype('float32'),
        y_train.astype('float32'),
        x_test.astype('float32'),
        y_test.astype('float32'),
        scaler,
    )


def build_keras_model(input_shape_tuple, optimizer_name='adam', learning_rate=0.001, l2_coeff=0.01):
    """Conv1D feature extraction -> stacked LSTM -> Dense regression head."""
    if optimizer_name.lower() == 'nadam':
        opt = Nadam(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    model = Sequential(name="Conv1D_LSTM_Model")
    model.add(Input(shape=input_shape_tuple, name="Input_Layer"))
    model.add(Conv1D(64, 5, padding='same', activation='relu',
                     kernel_initializer="glorot_uniform", name="Conv1D_1"))
    model.add(Dropout(0.1, name="Dropout_Conv"))
    model.add(Conv1D(32, 3, padding='same', activation='relu',
                     kernel_initializer="glorot_uniform", name="Conv1D_2"))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_coeff), name="LSTM_1"))
    model.add(Dropout(0.2, name="Dropout_1"))
    model.add(LSTM(32, return_sequences=True, kernel_regularizer=l2(l2_coeff), name="LSTM_2"))
    model.add(Dropout(0.25, name="Dropout_2"))
    model.add(LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_coeff), name="LSTM_3"))
    model.add(Dropout(0.3, name="Dropout_3"))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform", name="Dense_1"))
    model.add(Dropout(0.2, name="Dropout_4"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform", name="Output_Dense"))

    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    return model


def _evaluate_cv_model(model, scaler, X_test, y_test, df_historical, plot_path):
    """Evaluate the best CV model, plot actual vs predicted, print metrics."""
    loss_s, mae_s, _ = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Scaled Test Loss (MSE): {loss_s:.4f}")
    print(f"  Scaled Test MAE: {mae_s:.4f}")

    preds_scaled = model.predict(X_test).flatten()
    target_idx_in_scaled = COLUMNS_TO_SCALE.index(TARGET_COLUMN_NAME)

    temp_preds = np.zeros((len(preds_scaled), len(COLUMNS_TO_SCALE)))
    temp_preds[:, target_idx_in_scaled] = preds_scaled
    preds_original = scaler.inverse_transform(temp_preds)[:, target_idx_in_scaled]

    temp_actuals = np.zeros((len(y_test), len(COLUMNS_TO_SCALE)))
    temp_actuals[:, target_idx_in_scaled] = y_test
    actual_original = scaler.inverse_transform(temp_actuals)[:, target_idx_in_scaled]

    mae_orig = mean_absolute_error(actual_original, preds_original)
    mse_orig = mean_squared_error(actual_original, preds_original)
    rmse_orig = np.sqrt(mse_orig)
    print(f"  Original-scale MAE: {mae_orig:.2f}")
    print(f"  Original-scale MSE: {mse_orig:.2f}")
    print(f"  Original-scale RMSE: {rmse_orig:.2f}")

    plot_start = int(len(df_historical) * TRAIN_SPLIT_RATIO_CV) + N_PAST_STEPS
    plot_dates = df_historical.iloc[plot_start:plot_start + len(actual_original)].index
    plt.figure(figsize=(15, 6))
    plt.plot(plot_dates, actual_original, label='Actual (original scale)')
    plt.plot(plot_dates, preds_original, label='Predicted (original scale)', alpha=0.7)
    plt.title('CV Test Set: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(TARGET_COLUMN_NAME)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


def _build_future_frame(df_holidays, pred_dates, rollcap_file, month_sum_file):
    """Assemble future-side feature frame from exogenous CSVs + calendar."""
    df_future_exog = pd.read_csv(rollcap_file, parse_dates=['date'])

    df_future = pd.DataFrame({'date': pred_dates})
    df_future = pd.merge(df_future, df_future_exog[['date', 'RollCap']], on='date', how='left')
    df_future['RollCap'] = df_future['RollCap'].ffill().bfill().fillna(0)

    df_future = pd.merge(df_future, df_holidays, on='date', how='left')
    df_future['Holiday_A'] = df_future['Holiday_A'].fillna(0).astype(int)
    df_future['Holiday_B'] = df_future['Holiday_B'].fillna(0).astype(int)

    df_future = df_future.set_index(pd.to_datetime(df_future['date']))
    df_future['weekday'] = df_future.index.dayofweek
    df_future['week_number'] = df_future.index.isocalendar().week.astype(int)
    df_future['month'] = df_future.index.month

    try:
        df_future_ms = pd.read_csv(month_sum_file, parse_dates=['date'])
        df_future_ms['month_start_key'] = pd.to_datetime(df_future_ms['date']).dt.to_period('M').dt.to_timestamp()
        df_future_ms = df_future_ms.rename(columns={'monthly_sum': 'monthly_sum_future_val'})

        df_future['month_start_key'] = df_future.index.to_period('M').to_timestamp()
        df_future = df_future.reset_index(drop=True).merge(
            df_future_ms[['month_start_key', 'monthly_sum_future_val']],
            on='month_start_key', how='left',
        )
        df_future = df_future.rename(columns={'monthly_sum_future_val': 'monthly_sum'})
        df_future['monthly_sum'] = df_future['monthly_sum'].ffill().bfill().fillna(0)
        df_future = df_future.set_index(pd.to_datetime(df_future['date']))
        df_future.drop(columns=['month_start_key', 'date'], inplace=True, errors='ignore')
    except FileNotFoundError:
        print(f"Warning: {month_sum_file} not found; future monthly_sum will be 0.")
        df_future['monthly_sum'] = 0

    df_future[TARGET_COLUMN_NAME] = 0
    for col in FEATURE_ORDER:
        if col not in df_future.columns:
            df_future[col] = 0
    return df_future[list(FEATURE_ORDER)]


def _predict_future(model, scaler, df_historical, df_future_for_pred, pred_dates,
                    plot_path, csv_path):
    """Iteratively predict the future horizon one step at a time."""
    last_hist_window = df_historical[list(FEATURE_ORDER)].values[-N_PAST_STEPS:]
    combined_unscaled = np.concatenate((last_hist_window, df_future_for_pred.values))

    combined_df = pd.DataFrame(combined_unscaled, columns=list(FEATURE_ORDER))
    combined_scaled = combined_df.copy()
    combined_scaled[COLUMNS_TO_SCALE] = scaler.transform(combined_df[COLUMNS_TO_SCALE])
    input_array = combined_scaled.values.astype('float32')

    target_idx = FEATURE_ORDER.index(TARGET_COLUMN_NAME)
    preds_scaled = []
    for i in range(N_PAST_STEPS, len(input_array)):
        window = input_array[i - N_PAST_STEPS:i, :].reshape(1, N_PAST_STEPS, N_FEATURES)
        pred = model.predict(window, verbose=0)[0, 0]
        preds_scaled.append(pred)
        input_array[i, target_idx] = pred

    preds_np = np.array(preds_scaled).reshape(-1, 1)
    target_idx_in_scaled = COLUMNS_TO_SCALE.index(TARGET_COLUMN_NAME)
    dummy = np.zeros((len(preds_np), len(COLUMNS_TO_SCALE)))
    dummy[:, target_idx_in_scaled] = preds_np.flatten()
    preds_original = scaler.inverse_transform(dummy)[:, target_idx_in_scaled]
    preds_original_int = np.round(preds_original).astype(int)

    df_forecast = pd.DataFrame({
        'date': pred_dates,
        f'{TARGET_COLUMN_NAME}_forecast': preds_original_int,
    })
    print("\nFuture forecast (first 5 rows):")
    print(df_forecast.head())
    df_forecast.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
    print(f"Forecast saved to {csv_path}")

    plt.figure(figsize=(15, 7))
    plt.plot(df_historical.index[-60:], df_historical[TARGET_COLUMN_NAME].iloc[-60:],
             label='Historical actual')
    plt.plot(df_forecast['date'], df_forecast[f'{TARGET_COLUMN_NAME}_forecast'],
             label='Future forecast')
    plt.title('Forecast vs Historical')
    plt.xlabel('Date')
    plt.ylabel(TARGET_COLUMN_NAME)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")
    return df_forecast


def run_forecast_pipeline(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR, epochs=None):
    """Run the full pipeline: features -> CV -> final fit -> 12-month forecast.

    ``epochs`` overrides the single ``epochs`` value in GRID_SEARCH_PARAMS
    (useful for a quick CPU check); None keeps the configured value.
    """
    historical_data_file = os.path.join(data_dir, HISTORICAL_DATA_FILENAME)
    future_rollcap_file = os.path.join(data_dir, FUTURE_ROLLCAP_FILENAME)
    future_month_sum_file = os.path.join(data_dir, FUTURE_MONTH_SUM_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    scaler_cv_pkl = os.path.join(output_dir, SCALER_CV_FILENAME)
    scaler_final_pkl = os.path.join(output_dir, SCALER_FINAL_FILENAME)
    model_final_keras = os.path.join(output_dir, MODEL_FINAL_FILENAME)
    cv_test_plot = os.path.join(output_dir, CV_TEST_PLOT_FILENAME)
    forecast_plot = os.path.join(output_dir, FORECAST_PLOT_FILENAME)
    forecast_csv = os.path.join(output_dir, FORECAST_CSV_FILENAME)

    grid_search_params = dict(GRID_SEARCH_PARAMS)
    if epochs is not None:
        grid_search_params['epochs'] = [epochs]

    print("--- Starting Forecasting Pipeline ---")
    tf.keras.utils.set_random_seed(PIPELINE_SEED)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth.")
    else:
        print("No GPUs detected. Running on CPU.")

    print("\n[PHASE 1] Loading and preprocessing data...")
    df_holidays = load_holidays_data(HOLIDAY_YEARS, HOLIDAY_A_NAMES, US_HOLIDAYS_SUBSET)
    df_historical = engineer_features_historical(historical_data_file, df_holidays)
    print(f"Number of features: {N_FEATURES}. Feature order: {list(FEATURE_ORDER)}")

    if df_historical.isnull().values.any():
        print("WARNING: NaNs detected; filling with median / ffill / bfill.")
        for col in df_historical.columns:
            if df_historical[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_historical[col]):
                    df_historical[col] = df_historical[col].fillna(df_historical[col].median())
                else:
                    df_historical[col] = df_historical[col].fillna(df_historical[col].mode()[0])
        df_historical = df_historical.ffill().bfill().fillna(0)
        if df_historical.isnull().values.any():
            raise RuntimeError("NaNs persist after fill; aborting.")

    df_historical = inject_monthly_sum_noise(df_historical)
    print(f"Applied monthly_sum noise (std={MONTHLY_SUM_NOISE_STD}, seed={MONTHLY_SUM_NOISE_SEED}).")

    print("\n[PHASE 2] Scaling and creating sequences for CV...")
    X_train_cv, y_train_cv, X_test_cv, y_test_cv, scaler_cv_obj = scale_data_and_create_sequences(
        df_historical,
        TRAIN_SPLIT_RATIO_CV,
        N_PAST_STEPS,
        COLUMNS_TO_SCALE,
        FEATURE_ORDER,
        TARGET_COLUMN_NAME,
    )

    if X_train_cv.size == 0:
        raise RuntimeError("Not enough data to create X_train_cv sequences.")

    print(f"trainX_cv: {X_train_cv.shape}, trainY_cv: {y_train_cv.shape}")
    if X_test_cv.size:
        print(f"testX_cv: {X_test_cv.shape}, testY_cv: {y_test_cv.shape}")

    model_input_shape = (N_PAST_STEPS, N_FEATURES)

    print("\n[PHASE 3] Hyperparameter search (manual TimeSeriesSplit CV)...")
    try:
        with open(scaler_cv_pkl, 'wb') as f:
            pickle.dump(scaler_cv_obj, f)
        print(f"CV scaler saved to {scaler_cv_pkl}")
    except OSError as e:
        print(f"Warning: could not save CV scaler: {e}")

    combos = list(ParameterGrid(grid_search_params))
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    print(f"Evaluating {len(combos)} combos × {CV_N_SPLITS} folds = {len(combos) * CV_N_SPLITS} total fits")

    best_params = None
    best_cv_mse = np.inf
    for combo_idx, combo in enumerate(combos, 1):
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_cv), 1):
            tf.keras.backend.clear_session()
            model = build_keras_model(
                input_shape_tuple=model_input_shape,
                optimizer_name=combo['optimizer'],
                learning_rate=combo['learning_rate'],
                l2_coeff=combo['l2_coeff'],
            )
            model.fit(
                X_train_cv[train_idx], y_train_cv[train_idx],
                epochs=combo['epochs'],
                batch_size=combo['batch_size'],
                verbose=0,
                callbacks=[EarlyStopping(monitor='loss', patience=EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=0)],
            )
            preds = model.predict(X_train_cv[val_idx], verbose=0).flatten()
            fold_mse = mean_squared_error(y_train_cv[val_idx], preds)
            fold_scores.append(fold_mse)
            del model
        mean_mse = float(np.mean(fold_scores))
        print(f"  combo {combo_idx}/{len(combos)} {combo} -> mean_MSE={mean_mse:.5f}")
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_params = combo

    print(f"\nBest CV params: {best_params}")
    print(f"Best CV MSE (scaled): {best_cv_mse:.5f}")

    # Refit best params on the FULL X_train_cv so we have one model to evaluate
    # against X_test_cv (the held-out 5% tail).
    tf.keras.backend.clear_session()
    best_cv_model = build_keras_model(
        input_shape_tuple=model_input_shape,
        optimizer_name=best_params['optimizer'],
        learning_rate=best_params['learning_rate'],
        l2_coeff=best_params['l2_coeff'],
    )
    best_cv_model.fit(
        X_train_cv, y_train_cv,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0,
        callbacks=[EarlyStopping(monitor='loss', patience=EARLY_STOPPING_PATIENCE,
                                 restore_best_weights=True, verbose=0)],
    )
    best_cv_params = best_params

    if X_test_cv.size and y_test_cv.size:
        print("\nEvaluating best CV model on the held-out CV test set...")
        _evaluate_cv_model(best_cv_model, scaler_cv_obj, X_test_cv, y_test_cv, df_historical,
                           cv_test_plot)

    # --- Phase 4: Final model on all historical data ---
    print("\n[PHASE 4] Training final model on all historical data...")
    scaler_final_obj = MinMaxScaler(feature_range=(0, 1))
    df_historical_final_scaled = df_historical.copy()
    df_historical_final_scaled[COLUMNS_TO_SCALE] = scaler_final_obj.fit_transform(
        df_historical[COLUMNS_TO_SCALE]
    )

    try:
        with open(scaler_final_pkl, 'wb') as f:
            pickle.dump(scaler_final_obj, f)
        print(f"Final scaler saved to {scaler_final_pkl}")
    except OSError as e:
        print(f"Warning: could not save final scaler: {e}")

    X_all_hist, y_all_hist = create_sequences_from_data(
        df_historical_final_scaled[list(FEATURE_ORDER)].values,
        N_PAST_STEPS,
        N_FEATURES,
        FEATURE_ORDER.index(TARGET_COLUMN_NAME),
    )
    print(f"Full historical X: {X_all_hist.shape}, y: {y_all_hist.shape}")

    tf.keras.backend.clear_session()
    final_model = build_keras_model(
        input_shape_tuple=model_input_shape,
        optimizer_name=best_cv_params['optimizer'],
        learning_rate=best_cv_params['learning_rate'],
        l2_coeff=best_cv_params.get('l2_coeff', FIXED_L2_COEFF),
    )
    # Final training uses a 10% trailing validation slice for EarlyStopping —
    # Keras's validation_split takes the LAST fraction, which is time-ordered.
    early_stopping_final = EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True
    )
    final_model.fit(
        X_all_hist,
        y_all_hist,
        epochs=best_cv_params['epochs'],
        batch_size=best_cv_params['batch_size'],
        callbacks=[early_stopping_final],
        validation_split=0.1,
        verbose=1,
    )
    print("Final model training complete.")

    try:
        final_model.save(model_final_keras)
        print(f"Final model saved to {model_final_keras}")
        model_for_future_prediction = final_model
        scaler_for_future_prediction = scaler_final_obj
    except OSError as e:
        print(f"Warning: could not save final model: {e}. Falling back to CV-best model.")
        model_for_future_prediction = best_cv_model
        scaler_for_future_prediction = scaler_cv_obj

    # --- Phase 5: Future prediction ---
    print("\n[PHASE 5] Generating future predictions...")
    try:
        df_future_exog = pd.read_csv(future_rollcap_file, parse_dates=['date'])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Future RollCap file not found: {future_rollcap_file}. "
            "Run data/generate_rollcap_dummy_data.py first."
        )

    pred_start_date = df_historical.index.max() + pd.Timedelta(days=1)
    pred_dates = pd.date_range(start=pred_start_date, periods=len(df_future_exog))
    df_future_for_pred = _build_future_frame(
        df_holidays, pred_dates, future_rollcap_file, future_month_sum_file
    )

    _predict_future(
        model_for_future_prediction,
        scaler_for_future_prediction,
        df_historical,
        df_future_for_pred,
        pred_dates,
        forecast_plot,
        forecast_csv,
    )

    print("\n--- Forecasting Pipeline Finished ---")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the Conv1D + LSTM model and write a 12-month daily forecast.",
    )
    parser.add_argument(
        '--data-dir', default=DEFAULT_DATA_DIR,
        help="Directory holding the three input CSVs written by data/generate_*.py "
             "(default: %(default)s).",
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help="Where to write the model, scalers, plots and forecast CSV (default: %(default)s).",
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Override the training epochs in GRID_SEARCH_PARAMS "
             f"(default: {GRID_SEARCH_PARAMS['epochs'][0]}). Use a small value for a quick check.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_forecast_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, epochs=args.epochs)


if __name__ == '__main__':
    main()
