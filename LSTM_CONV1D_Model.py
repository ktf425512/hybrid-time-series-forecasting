# --- Standard Library Imports ---
import os
import traceback
import pickle

# --- Third-party Library Imports ---
import holidays
import matplotlib.dates as mdates # Not explicitly used in the direct flow, but kept from notebook
import numpy as np
import pandas as pd
# import seaborn as sns # Not critically used in the core notebook logic for this script
import tensorflow as tf

from matplotlib import pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (LSTM, Conv1D, Dense, Activation,
                                     Dropout, Input, Flatten) # Added Flatten for simple model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2

# --- Configuration Parameters ---
# Determine the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths (adjust if your files are elsewhere relative to the script)
HISTORICAL_DATA_FILE = os.path.join(BASE_DIR, 'dummy_data.csv')
FUTURE_ROLLCAP_FILE = os.path.join(BASE_DIR, 'mock_future_RollCap.csv')
FUTURE_MONTH_SUM_FILE = os.path.join(BASE_DIR, 'future_Month_Sum.csv')

# Output files
SCALER_CV_PKL = os.path.join(BASE_DIR, 'scaler_cv_notebook_converted.pkl')
SCALER_FINAL_PKL = os.path.join(BASE_DIR, 'scaler_final_notebook_converted.pkl')
MODEL_FINAL_KERAS = os.path.join(BASE_DIR, 'model_final_notebook_converted.keras')

# Holiday Settings
HOLIDAY_YEARS = [2021, 2022, 2023, 2024, 2025, 2026, 2027] # Extend as needed
US_HOLIDAYS_SUBSET = [
    "New Year's Day", "New Year's Day (observed)", "Memorial Day",
    "Independence Day", "Labor Day", "Thanksgiving", "Christmas Day"
]
HOLIDAY_A_NAMES = ["Thanksgiving", "Christmas Day"]

# Feature Engineering & Model Structure
N_PAST_STEPS = 7  # Look-back window size
# N_FEATURES_DYNAMIC will be determined dynamically after all features are created.
TARGET_COLUMN_NAME = 'Total_Presented'
COLUMNS_TO_SCALE = ['Total_Presented', 'RollCap', 'monthly_sum']
# ALL_FEATURES_ORDERED_LIST will also be determined dynamically.

# Training Settings
TRAIN_SPLIT_RATIO_CV = 0.95 # For GridSearchCV and its internal test split

# GridSearchCV Settings
CV_N_SPLITS = 3
GRID_SEARCH_PARAMS = {
    'batch_size': [16,32],
    'epochs': [30, 50], 
    'optimizer': ['adam', 'nadam'],
    'optimizer__learning_rate': [0.0005, 0.001],
    'model__l2_coeff': [0.01, 0.001] 
}
EARLY_STOPPING_PATIENCE = 10
FIXED_L2_COEFF = 0.01 # Used if not tuning l2_coeff via GridSearchCV

# --- Helper Functions ---

def load_holidays_data(years, holiday_a_names_list, us_holidays_subset_list):
    # Loads and preprocesses holiday data, creating Holiday_A and Holiday_B features.
    h_list = []
    for item in holidays.UnitedStates(years=years).items():
        h_list.append(item)
    
    df_h = pd.DataFrame(h_list, columns=['date', 'holiday_name'])
    df_h['date'] = pd.to_datetime(df_h['date'])
    df_h.drop_duplicates(subset=['date'], keep='first', inplace=True)
    
    df_h['Holiday_A'] = df_h['holiday_name'].apply(lambda x: 1 if x in holiday_a_names_list else 0)
    df_h['Holiday_B'] = df_h['holiday_name'].apply(
        lambda x: 1 if (x in us_holidays_subset_list and x not in holiday_a_names_list) else 0
    )
    return df_h[['date', 'Holiday_A', 'Holiday_B']]

def engineer_features_historical(filepath, df_holidays):
    # Loads historical data and engineers all features including monthly_sum.
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    df = pd.merge(df, df_holidays, on='date', how='left')
    df['Holiday_A'] = df['Holiday_A'].fillna(0).astype(int)
    df['Holiday_B'] = df['Holiday_B'].fillna(0).astype(int)
    
    df['date_dt'] = pd.to_datetime(df['date']) 
    
    df['month_start_key'] = df['date_dt'].dt.to_period('M').apply(lambda p: p.to_timestamp())

    df['weekday'] = df['date_dt'].dt.dayofweek
    df['week_number'] = df['date_dt'].dt.isocalendar().week.astype(int)
    df['month'] = df['date_dt'].dt.month
    
    df_temp_for_monthly_sum = df.set_index('date_dt') 
    monthly_sum_series = df_temp_for_monthly_sum[TARGET_COLUMN_NAME].resample('MS').sum()
    monthly_sum_series.name = 'monthly_sum_val' 
    
    df = pd.merge(df, monthly_sum_series, left_on='month_start_key', right_index=True, how='left')
    df.rename(columns={'monthly_sum_val': 'monthly_sum'}, inplace=True)
    df['monthly_sum'] = df['monthly_sum'].ffill().bfill().fillna(0) 
    
    df = df.set_index('date_dt') 
    
    feature_order = [
        TARGET_COLUMN_NAME, 'RollCap', 'Holiday_A', 'Holiday_B',
        'weekday', 'week_number', 'month', 'monthly_sum'
    ]
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0 
    df = df[feature_order]
    
    cols_to_drop = ['date', 'month_start_key'] 
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

    return df, feature_order

def scale_data_and_create_sequences(df_full, train_ratio, n_past, cols_to_scale_list, 
                                    ordered_features_list, target_col_name):
    # Splits, scales, and creates sequences for training and testing.
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
    else:
        print("Warning: Test set is empty, skipping scaling for it.")

    df_train_scaled_ordered_values = df_train_scaled[ordered_features_list].values
    df_test_scaled_ordered_values = df_test_scaled[ordered_features_list].values
    
    n_features = len(ordered_features_list)
    target_col_idx_in_ordered = ordered_features_list.index(target_col_name)
    
    x_train, y_train = create_sequences_from_data(df_train_scaled_ordered_values, n_past, n_features, target_col_idx_in_ordered)
    x_test, y_test = create_sequences_from_data(df_test_scaled_ordered_values, n_past, n_features, target_col_idx_in_ordered)
    
    # Ensure float32 type for Keras
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    
    return x_train, y_train, x_test, y_test, scaler

def create_sequences_from_data(dataset_values, n_past, n_features_expected, target_col_idx):
    # Creates X, Y sequences from numpy array data.
    data_x, data_y = [], []
    if len(dataset_values) <= n_past:
        print(f"Warning: Dataset length {len(dataset_values)} too short for n_past={n_past}")
        return np.array(data_x), np.array(data_y) # Will be converted to float32 by caller

    if dataset_values.shape[1] != n_features_expected:
         raise ValueError(f"Dataset has {dataset_values.shape[1]} features, expected {n_features_expected}")

    for i in range(n_past, len(dataset_values)):
        data_x.append(dataset_values[i - n_past:i, :])
        data_y.append(dataset_values[i, target_col_idx])
    return np.array(data_x), np.array(data_y) # Caller will astype to float32

# --- Model Building Functions ---
def build_keras_model_ULTRA_SIMPLE(input_shape_tuple_param):
    """Builds an extremely simple Keras model for debugging purposes."""
    print(f"[DEBUG SimpleModel] Building ULTRA_SIMPLE model with input shape: {input_shape_tuple_param}")
    model = Sequential(name="UltraSimpleModel_Debug")
    model.add(Input(shape=input_shape_tuple_param))
    model.add(Flatten()) 
    model.add(Dense(1, activation="linear")) 
    
    optimizer = Adam(learning_rate=0.001) 
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    print("[DEBUG SimpleModel] ULTRA_SIMPLE model compiled.")
    model.summary(print_fn=lambda x: print(f"[DEBUG SimpleModel] {x}")) 
    return model

def build_keras_model(input_shape_tuple, optimizer_name='adam', learning_rate=0.001, l2_coeff=0.01):
    """Builds the Keras Conv1D + LSTM model."""
    if optimizer_name.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'nadam':
        opt = Nadam(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate) 

    model = Sequential(name="Conv1D_LSTM_Model_Script_Refactored")
    model.add(Input(shape=input_shape_tuple, name="Input_Layer"))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', kernel_initializer="glorot_uniform", name="Conv1D_1"))
    model.add(Dropout(0.1, name="Dropout_Conv"))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer="glorot_uniform", name="Conv1D_2"))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_coeff), name="LSTM_1"))
    model.add(Dropout(0.2, name="Dropout_1"))
    model.add(LSTM(32, return_sequences=True, kernel_regularizer=l2(l2_coeff), name="LSTM_2"))
    model.add(Dropout(0.25, name="Dropout_2"))
    model.add(LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_coeff), name="LSTM_3"))
    model.add(Dropout(0.3, name="Dropout_3"))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform", name="Dense_1"))
    model.add(Dropout(0.2, name="Dropout_4"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform", name="Output_Dense"))

    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])
    return model

# --- Main Script Execution ---
def run_forecast_pipeline():
    """Main function to run the entire forecasting pipeline."""
    print("--- Starting Forecasting Pipeline ---")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Successfully configured {len(gpus)} GPUs with memory growth.")
        except RuntimeError as e:
            print(f"Error during GPU memory growth configuration: {e}")
    else:
        print("No GPUs detected by TensorFlow. Running on CPU.")

    print("\n[PHASE 1] Loading and Preprocessing Data...")
    df_holidays_processed = load_holidays_data(HOLIDAY_YEARS, HOLIDAY_A_NAMES, US_HOLIDAYS_SUBSET)
    df_historical, ALL_FEATURES_ORDERED_LIST = engineer_features_historical(HISTORICAL_DATA_FILE, df_holidays_processed)
    
    N_FEATURES_DYNAMIC = len(ALL_FEATURES_ORDERED_LIST)
    print(f"Data loaded. Number of features: {N_FEATURES_DYNAMIC}. Feature order: {ALL_FEATURES_ORDERED_LIST}")

    if df_historical.isnull().values.any():
        print("\nWARNING: NaNs detected in historical data after feature engineering!")
        print(df_historical.isnull().sum())
        print("Attempting to fill NaNs with median/mode or ffill/bfill. Review carefully.")
        for col in df_historical.columns:
            if df_historical[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_historical[col]):
                    df_historical[col] = df_historical[col].fillna(df_historical[col].median())
                else: 
                    df_historical[col] = df_historical[col].fillna(df_historical[col].mode()[0]) 
        df_historical = df_historical.ffill().bfill().fillna(0)
        if df_historical.isnull().values.any():
            print("FATAL: NaNs persist after fill attempts. Exiting.")
            return
        else:
            print("NaNs filled.")

    print("\n[PHASE 2] Data Prep for CV and Direct Fit Debug...") # Renamed phase slightly
    X_train_cv, y_train_cv, X_test_cv, y_test_cv, scaler_cv_obj = scale_data_and_create_sequences(
        df_historical,
        TRAIN_SPLIT_RATIO_CV,
        N_PAST_STEPS,
        COLUMNS_TO_SCALE,
        ALL_FEATURES_ORDERED_LIST,
        TARGET_COLUMN_NAME
    )

    if X_train_cv.size == 0 : # Simplified check
        print("ERROR: Not enough data to create X_train_cv sequences. Exiting.")
        return
        
    print(f"trainX_cv shape: {X_train_cv.shape}, trainY_cv shape: {y_train_cv.shape}")
    if X_test_cv.size > 0:
        print(f"testX_cv shape: {X_test_cv.shape}, testY_cv shape: {y_test_cv.shape}")
    else:
        print("testX_cv or y_test_cv is empty.")

    current_model_input_shape = (N_PAST_STEPS, N_FEATURES_DYNAMIC) 
    print(f"[DEBUG] current_model_input_shape in run_forecast_pipeline defined as: {current_model_input_shape}") 
    
    # --- Temporary Debug Block for direct model.fit with ULTRA_SIMPLE model ---
    print("\n[DEBUG] Attempting direct Keras model.fit() WITH ULTRA_SIMPLE_MODEL...")
    direct_fit_successful = False
    if X_train_cv.size > 0:
        try:
            debug_model_instance = build_keras_model_ULTRA_SIMPLE( 
                input_shape_tuple_param=current_model_input_shape
            )
            
            print("[DEBUG] Starting direct model.fit() for 3 epochs with ULTRA_SIMPLE model...")
            history_debug = debug_model_instance.fit( 
                X_train_cv,
                y_train_cv,
                epochs=3,       
                batch_size=32,  
                verbose=1       
            )
            print("[DEBUG] Direct Keras model.fit() with ULTRA_SIMPLE model completed 3 epochs.")
            print(f"[DEBUG] History from direct fit: {history_debug.history}")
            direct_fit_successful = True 
        except Exception as e_debug_fit:
            print(f"[DEBUG] ERROR during direct Keras model.fit() with ULTRA_SIMPLE model: {e_debug_fit}")
            traceback.print_exc()
            direct_fit_successful = False
    else:
        print("[DEBUG] Skipping direct model fit as X_train_cv is empty.")
    print("[DEBUG] Finished direct Keras model.fit() attempt with ULTRA_SIMPLE model.\n")
    
    # --- If direct fit was successful, then proceed to GridSearchCV with the original complex model ---
    # --- Otherwise, we might skip GridSearchCV or try it with the simple model for further diagnosis ---
    if not direct_fit_successful:
        print("[CRITICAL DEBUG] Direct fit with ULTRA_SIMPLE model failed or was skipped. ")
        print("Skipping GridSearchCV with the complex model as a basic fit is not working.")
        print("Please check RAM usage during the direct fit and TensorFlow/environment integrity.")
        # Optionally, you could try GridSearchCV with the ULTRA_SIMPLE model here as a further test:
        # print("[DEBUG] Trying GridSearchCV with ULTRA_SIMPLE model...")
        # simple_estimator = KerasRegressor(model=build_keras_model_ULTRA_SIMPLE, model__input_shape_tuple_param=current_model_input_shape, verbose=1)
        # simple_grid_params = {'batch_size': [32], 'epochs': [2]} # minimal params
        # tscv_simple = TimeSeriesSplit(n_splits=2)
        # grid_simple = GridSearchCV(estimator=simple_estimator, param_grid=simple_grid_params, cv=tscv_simple, verbose=2, n_jobs=1, error_score='raise')
        # try:
        #     grid_simple.fit(X_train_cv, y_train_cv)
        #     print("[DEBUG] GridSearchCV with ULTRA_SIMPLE model completed.")
        # except Exception as e_grid_simple:
        #     print(f"[DEBUG] Error with GridSearchCV and ULTRA_SIMPLE model: {e_grid_simple}")
        #     traceback.print_exc()
        # return # End execution here if direct fit failed
    
    # --- Original GridSearchCV Phase (Now called Phase 3) ---
    print("\n[PHASE 3] Hyperparameter Tuning with GridSearchCV (Using original complex model)...")
    try:
        with open(SCALER_CV_PKL, 'wb') as f: # Save scaler before potential crash in GridSearchCV
            pickle.dump(scaler_cv_obj, f)
        print(f"CV Scaler saved to {SCALER_CV_PKL}")
    except Exception as e:
        print(f"Error saving CV scaler: {e}")

    keras_estimator_cv = KerasRegressor(
        model=build_keras_model, # Using the original complex model builder
        model__input_shape_tuple=current_model_input_shape, 
        loss="mse", 
        verbose=1, # INCREASED Keras verbosity during fit
    )

    tscv_cv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    
    grid_search_obj = GridSearchCV(
        estimator=keras_estimator_cv,
        param_grid=GRID_SEARCH_PARAMS,
        cv=tscv_cv,
        scoring='neg_mean_squared_error',
        verbose=2,
        error_score='raise'
    )

    early_stopping_callback_cv = EarlyStopping(
        monitor='loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True
    )

    best_cv_model = None
    best_cv_params = None

    print("\nStarting GridSearchCV fit (with complex model)...")
    try:
        grid_search_obj.fit(X_train_cv, y_train_cv, callbacks=[early_stopping_callback_cv])
        print("\nGridSearchCV completed!")
        print(f"Best parameters found: {grid_search_obj.best_params_}")
        print(f"Best CV score (neg_mean_squared_error): {grid_search_obj.best_score_:.4f}")
        
        best_cv_model = grid_search_obj.best_estimator_.model_
        best_cv_params = grid_search_obj.best_params_

        if X_test_cv.size > 0 and y_test_cv.size > 0:
            print("\nEvaluating best model from CV on the CV test set...")
            # ... (evaluation logic from your script, remains the same) ...
            loss_s, mae_s, mse_s = best_cv_model.evaluate(X_test_cv, y_test_cv, verbose=0)
            print(f"  Scaled Test Loss (MSE from model): {loss_s:.4f}")
            print(f"  Scaled Test MAE: {mae_s:.4f}")

            preds_scaled_cv = best_cv_model.predict(X_test_cv).flatten()
            
            temp_preds = np.zeros((len(preds_scaled_cv), len(COLUMNS_TO_SCALE)))
            target_idx_in_scaled_cols = COLUMNS_TO_SCALE.index(TARGET_COLUMN_NAME)
            temp_preds[:, target_idx_in_scaled_cols] = preds_scaled_cv
            preds_original_cv = scaler_cv_obj.inverse_transform(temp_preds)[:, target_idx_in_scaled_cols]

            temp_actuals = np.zeros((len(y_test_cv), len(COLUMNS_TO_SCALE)))
            temp_actuals[:, target_idx_in_scaled_cols] = y_test_cv
            actual_original_cv = scaler_cv_obj.inverse_transform(temp_actuals)[:, target_idx_in_scaled_cols]

            mae_orig = mean_absolute_error(actual_original_cv, preds_original_cv)
            mse_orig = mean_squared_error(actual_original_cv, preds_original_cv)
            rmse_orig = np.sqrt(mse_orig)
            print(f"  Original Scale Test MAE: {mae_orig:.2f}")
            print(f"  Original Scale Test MSE: {mse_orig:.2f}")
            print(f"  Original Scale Test RMSE: {rmse_orig:.2f}")

            plt.figure(figsize=(15, 6))
            plot_dates_cv_test = df_historical.iloc[int(len(df_historical) * TRAIN_SPLIT_RATIO_CV):].index[N_PAST_STEPS:N_PAST_STEPS+len(actual_original_cv)]
            plt.plot(plot_dates_cv_test, actual_original_cv, label='Actual (Original Scale)')
            plt.plot(plot_dates_cv_test, preds_original_cv, label='Predicted (Original Scale)', alpha=0.7)
            plt.title('CV Test Set: Actual vs. Predicted (Original Scale)')
            plt.xlabel('Date'); plt.ylabel(TARGET_COLUMN_NAME); plt.legend()
            plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    except Exception as e:
        print(f"An error occurred during GridSearchCV or evaluation: {e}")
        traceback.print_exc()
    
    # --- Phase 4: (Optional) Train Final Model on All Historical Data ---
    model_for_future_prediction = None
    scaler_for_future_prediction = None # Initialize

    if best_cv_params:
        print("\n[PHASE 4] Training Final Model on All Historical Data...")
        # ... (logic for training final model, remains largely the same) ...
        # ... ensure scaler_final_obj is defined and used ...
        scaler_final_obj = MinMaxScaler(feature_range=(0,1)) # Define it here
        df_historical_final_scaled = df_historical.copy()
        df_historical_final_scaled[COLUMNS_TO_SCALE] = scaler_final_obj.fit_transform(df_historical[COLUMNS_TO_SCALE])
        
        try:
            with open(SCALER_FINAL_PKL, 'wb') as f: pickle.dump(scaler_final_obj, f)
            print(f"Final scaler saved to {SCALER_FINAL_PKL}")
        except Exception as e: print(f"Error saving final scaler: {e}")

        X_all_hist, y_all_hist = create_sequences_from_data(
            df_historical_final_scaled[ALL_FEATURES_ORDERED_LIST].values,
            N_PAST_STEPS, N_FEATURES_DYNAMIC, 
            ALL_FEATURES_ORDERED_LIST.index(TARGET_COLUMN_NAME)
        )
        print(f"Full historical X shape: {X_all_hist.shape}, Y shape: {y_all_hist.shape}")

        if X_all_hist.size > 0:
            print("Building and training the final model with best CV parameters...")
            final_model = build_keras_model( # Use the original complex model
                input_shape_tuple=current_model_input_shape, 
                optimizer_name=best_cv_params['optimizer'],
                learning_rate=best_cv_params['optimizer__learning_rate'],
                l2_coeff=best_cv_params.get('model__l2_coeff', FIXED_L2_COEFF) 
            )
            # ... (rest of final model training) ...
            early_stopping_final_train = EarlyStopping(
                monitor='loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True
            )
            final_model.fit(
                X_all_hist, y_all_hist,
                epochs=best_cv_params['epochs'],
                batch_size=best_cv_params['batch_size'],
                callbacks=[early_stopping_final_train],
                verbose=1
            )
            print("Final model training complete.")
            try:
                final_model.save(MODEL_FINAL_KERAS)
                print(f"Final model saved to {MODEL_FINAL_KERAS}")
                model_for_future_prediction = final_model
                scaler_for_future_prediction = scaler_final_obj
            except Exception as e: print(f"Error saving final model: {e}")
        else:
            print("Skipping final model training: No sequences from full historical data.")
    else:
        print("\nSkipping final model training: Best parameters from CV not available.")

    if model_for_future_prediction is None and best_cv_model is not None:
        print("Using best model from CV for future predictions as final model training was skipped or failed.")
        model_for_future_prediction = best_cv_model
        scaler_for_future_prediction = scaler_cv_obj 
    elif model_for_future_prediction is None: 
        if os.path.exists(MODEL_FINAL_KERAS) and os.path.exists(SCALER_FINAL_PKL):
            try:
                print(f"Loading production model from {MODEL_FINAL_KERAS} and scaler from {SCALER_FINAL_PKL}")
                model_for_future_prediction = tf.keras.models.load_model(MODEL_FINAL_KERAS)
                with open(SCALER_FINAL_PKL, 'rb') as f:
                    scaler_for_future_prediction = pickle.load(f)
                print("Loaded production model and scaler successfully.")
            except Exception as e:
                print(f"Error loading production model/scaler: {e}")
        else:
            print("No model available for prediction (final training skipped/failed, and no saved model found).")

    # --- Phase 5: Future Prediction ---
    if model_for_future_prediction and scaler_for_future_prediction:
        print("\n[PHASE 5] Generating Future Predictions...")
        # ... (future prediction logic, remains largely the same) ...
        print("Loading and preparing future exogenous data...")
        try:
            df_future_exog = pd.read_csv(FUTURE_ROLLCAP_FILE, parse_dates=['date'])
        except FileNotFoundError:
            print(f"ERROR: Future RollCap file not found: {FUTURE_ROLLCAP_FILE}. Cannot make future predictions.")
            return

        pred_start_date = df_historical.index.max() + pd.Timedelta(days=1)
        num_pred_periods = len(df_future_exog) 
        pred_dates = pd.date_range(start=pred_start_date, periods=num_pred_periods)

        df_future_base = pd.DataFrame({'date': pred_dates})
        df_future_base = pd.merge(df_future_base, df_future_exog[['date', 'RollCap']], on='date', how='left')
        df_future_base['RollCap'] = df_future_base['RollCap'].ffill().bfill().fillna(0) 

        df_future_base = pd.merge(df_future_base, df_holidays_processed, on='date', how='left')
        df_future_base['Holiday_A'] = df_future_base['Holiday_A'].fillna(0).astype(int)
        df_future_base['Holiday_B'] = df_future_base['Holiday_B'].fillna(0).astype(int)
        
        df_future_base['date_dt_temp'] = pd.to_datetime(df_future_base['date']) 
        df_future_base = df_future_base.set_index('date_dt_temp')
        df_future_base['weekday'] = df_future_base.index.dayofweek
        df_future_base['week_number'] = df_future_base.index.isocalendar().week.astype(int)
        df_future_base['month'] = df_future_base.index.month
        
        try:
            df_future_monthly_sums = pd.read_csv(FUTURE_MONTH_SUM_FILE, parse_dates=['date'])
            df_future_monthly_sums.rename(columns={'date': 'month_start_key', 'monthly_sum': 'monthly_sum_future_val'}, inplace=True)
            df_future_monthly_sums['month_start_key'] = pd.to_datetime(df_future_monthly_sums['month_start_key']).dt.to_period('M').to_timestamp()

            df_future_base['month_start_key'] = df_future_base.index.to_period('M').to_timestamp()
            df_future_base = pd.merge(df_future_base.reset_index(), 
                                    df_future_monthly_sums[['month_start_key', 'monthly_sum_future_val']],
                                    on='month_start_key',
                                    how='left')
            df_future_base.rename(columns={'monthly_sum_future_val': 'monthly_sum'}, inplace=True)
            df_future_base['monthly_sum'] = df_future_base['monthly_sum'].ffill().bfill().fillna(0)
            df_future_base = df_future_base.set_index('date_dt_temp') 
            df_future_base.drop(columns=['month_start_key', 'date'], errors='ignore', inplace=True)

        except FileNotFoundError:
            print(f"Warning: {FUTURE_MONTH_SUM_FILE} not found. Future 'monthly_sum' will be 0.")
            df_future_base['monthly_sum'] = 0
        except Exception as e:
            print(f"Error processing future monthly sums: {e}. Future 'monthly_sum' will be 0.")
            df_future_base['monthly_sum'] = 0
            
        df_future_base[TARGET_COLUMN_NAME] = 0 

        for col in ALL_FEATURES_ORDERED_LIST:
            if col not in df_future_base.columns:
                df_future_base[col] = 0
        df_future_for_pred = df_future_base[ALL_FEATURES_ORDERED_LIST]

        df_historical_ordered = df_historical[ALL_FEATURES_ORDERED_LIST]
        last_hist_window_unscaled = df_historical_ordered.values[-N_PAST_STEPS:]
        
        combined_unscaled_for_pred_loop = np.concatenate(
            (last_hist_window_unscaled, df_future_for_pred.values)
        )
        
        combined_df_for_pred_loop = pd.DataFrame(combined_unscaled_for_pred_loop, columns=ALL_FEATURES_ORDERED_LIST)
        combined_df_for_pred_loop_scaled = combined_df_for_pred_loop.copy()
        combined_df_for_pred_loop_scaled[COLUMNS_TO_SCALE] = scaler_for_future_prediction.transform(
            combined_df_for_pred_loop[COLUMNS_TO_SCALE]
        )
        
        input_array_for_pred_loop = combined_df_for_pred_loop_scaled.values.astype('float32')
        
        print("Starting iterative prediction for future dates...")
        all_future_preds_scaled = []
        
        for i in range(N_PAST_STEPS, len(input_array_for_pred_loop)): 
            current_sequence = input_array_for_pred_loop[i-N_PAST_STEPS:i, :].reshape(1, N_PAST_STEPS, N_FEATURES_DYNAMIC)
            predicted_value_scaled = model_for_future_prediction.predict(current_sequence, verbose=0)[0,0]
            all_future_preds_scaled.append(predicted_value_scaled)
            
            if i < len(input_array_for_pred_loop): 
                 input_array_for_pred_loop[i, ALL_FEATURES_ORDERED_LIST.index(TARGET_COLUMN_NAME)] = predicted_value_scaled
        
        all_future_preds_scaled_np = np.array(all_future_preds_scaled).reshape(-1, 1)
        
        dummy_for_inverse = np.zeros((len(all_future_preds_scaled_np), len(COLUMNS_TO_SCALE)))
        target_idx_in_scaled_cols = COLUMNS_TO_SCALE.index(TARGET_COLUMN_NAME)
        dummy_for_inverse[:, target_idx_in_scaled_cols] = all_future_preds_scaled_np.flatten()
        
        future_preds_original = scaler_for_future_prediction.inverse_transform(dummy_for_inverse)[:, target_idx_in_scaled_cols]
        future_preds_original_int = np.round(future_preds_original).astype(int)
        
        df_forecast_results = pd.DataFrame({'date': pred_dates, f'{TARGET_COLUMN_NAME}_forecast': future_preds_original_int})
        print("\nFuture Forecast Results:")
        print(df_forecast_results.head())

        plt.figure(figsize=(15,7))
        plt.plot(df_historical.index[-60:], df_historical[TARGET_COLUMN_NAME].iloc[-60:], label='Historical Actual')
        plt.plot(df_forecast_results['date'], df_forecast_results[f'{TARGET_COLUMN_NAME}_forecast'], label='Future Forecast')
        plt.title('Forecast vs Historical'); plt.xlabel('Date'); plt.ylabel(TARGET_COLUMN_NAME); plt.legend()
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()
    else:
        print("\nSkipping Future Prediction phase as no model is available.")

    print("\n--- Forecasting Pipeline Finished ---")

# --- Script Entry Point ---
if __name__ == '__main__':
    # Optional: Set TF_ENABLE_ONEDNN_OPTS before importing TensorFlow if needed
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
    run_forecast_pipeline()