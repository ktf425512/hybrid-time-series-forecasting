"""End-to-end smoke test: tiny synthetic inputs, one epoch on CPU.

Checks that the pipeline runs from CSV inputs to forecast outputs without
error. It makes no claim about forecast quality.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LSTM_CONV1D_Model as pipeline  # noqa: E402

N_HISTORY_DAYS = 200   # 5% test tail must exceed N_PAST_STEPS to be evaluated
N_FUTURE_DAYS = 14


@pytest.fixture
def tiny_data_dir(tmp_path):
    rng = np.random.default_rng(0)
    hist_dates = pd.date_range('2024-01-01', periods=N_HISTORY_DAYS, freq='D')
    pd.DataFrame({
        'date': hist_dates,
        'Total_Presented': rng.integers(80_000, 100_000, size=N_HISTORY_DAYS),
        'capacity': rng.integers(20_000, 30_000, size=N_HISTORY_DAYS),
        'RollCap': rng.integers(600_000, 750_000, size=N_HISTORY_DAYS),
    }).to_csv(tmp_path / pipeline.HISTORICAL_DATA_FILENAME, index=False)

    future_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=N_FUTURE_DAYS)
    pd.DataFrame({
        'date': future_dates,
        'RollCap': rng.integers(600_000, 750_000, size=N_FUTURE_DAYS),
    }).to_csv(tmp_path / pipeline.FUTURE_ROLLCAP_FILENAME, index=False)

    month_starts = future_dates.to_period('M').unique().to_timestamp()
    pd.DataFrame({
        'date': month_starts,
        'monthly_sum': [2_700_000] * len(month_starts),
    }).to_csv(tmp_path / pipeline.FUTURE_MONTH_SUM_FILENAME, index=False)
    return tmp_path


def test_pipeline_runs_end_to_end(tiny_data_dir, tmp_path):
    output_dir = tmp_path / 'outputs'

    pipeline.run_forecast_pipeline(data_dir=str(tiny_data_dir), output_dir=str(output_dir), epochs=1)

    for name in (
        pipeline.MODEL_FINAL_FILENAME,
        pipeline.SCALER_CV_FILENAME,
        pipeline.SCALER_FINAL_FILENAME,
        pipeline.CV_TEST_PLOT_FILENAME,
        pipeline.FORECAST_PLOT_FILENAME,
        pipeline.FORECAST_CSV_FILENAME,
    ):
        assert (output_dir / name).is_file(), name

    forecast = pd.read_csv(output_dir / pipeline.FORECAST_CSV_FILENAME)
    assert len(forecast) == N_FUTURE_DAYS
    assert forecast['Total_Presented_forecast'].notna().all()
