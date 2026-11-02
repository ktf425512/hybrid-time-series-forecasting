"""Shared simulation constants and helpers for the three data generators.

All three `generate_*.py` scripts previously held private copies of these
numbers. Keeping one authoritative definition here avoids silent drift when
one script is tweaked and the others aren't.

The helpers preserve the RNG-draw order of the original scripts so the
generated CSVs remain bit-identical after the refactor.
"""
import numpy as np
import pandas as pd

# --- Total_Presented (calls) generation ---
BASE_CALLS = 90_000
CALLS_AMPLITUDE_YEAR = 10_000
CALLS_NOISE_STD = 2_000
YEARLY_TREND_PHASE_SHIFT = np.pi / 4
WEEKDAY_FACTORS_CALLS = np.array([1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.85])  # Mon..Sun
MIN_TOTAL_PRESENTED = 80_001

# --- Capacity generation ---
BASE_CAPACITY = 25_000
CAPACITY_AMPLITUDE_YEAR = 5_000
CAPACITY_NOISE_STD = 1_000
WEEKDAY_FACTORS_CAP = np.array([1.05, 1.05, 1.0, 0.95, 0.9, 0.85, 0.9])  # Mon..Sun
MIN_CAPACITY = 20_001

# --- Shared ---
DEFAULT_SEED = 42
ROLLCAP_WINDOW = 30


def _as_datetime_index(dates):
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    return pd.DatetimeIndex(dates)


def _capacity_from_noise(dates, noise):
    idx = _as_datetime_index(dates)
    yearly = BASE_CAPACITY + CAPACITY_AMPLITUDE_YEAR * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    weekly = WEEKDAY_FACTORS_CAP[idx.dayofweek] * (BASE_CAPACITY / WEEKDAY_FACTORS_CAP.mean())
    raw = yearly * 0.5 + weekly * 0.5 + noise
    return np.maximum(MIN_CAPACITY, raw).astype(int)


def _total_presented_from_noise(dates, noise):
    idx = _as_datetime_index(dates)
    yearly = BASE_CALLS + CALLS_AMPLITUDE_YEAR * np.sin(
        2 * np.pi * idx.dayofyear / 365.25 + YEARLY_TREND_PHASE_SHIFT
    )
    weekly = WEEKDAY_FACTORS_CALLS[idx.dayofweek] * (BASE_CALLS / WEEKDAY_FACTORS_CALLS.mean())
    raw = yearly * 0.4 + weekly * 0.6 + noise
    return np.maximum(MIN_TOTAL_PRESENTED, raw).astype(int)


def simulate_daily_capacity(dates, seed=DEFAULT_SEED):
    """Standalone capacity simulation. Resets the global RNG."""
    np.random.seed(seed)
    noise = np.random.normal(0, CAPACITY_NOISE_STD, size=len(dates))
    return _capacity_from_noise(dates, noise)


def simulate_daily_total_presented(dates, seed=DEFAULT_SEED):
    """Standalone Total_Presented simulation. Resets the global RNG."""
    np.random.seed(seed)
    noise = np.random.normal(0, CALLS_NOISE_STD, size=len(dates))
    return _total_presented_from_noise(dates, noise)


def simulate_daily_capacity_and_total_presented(dates, seed=DEFAULT_SEED):
    """Combined simulation with a single shared RNG stream.

    Draws capacity noise first, then Total_Presented noise — matching the
    original `generate_dummy_data.py` order so outputs stay bit-identical.
    """
    np.random.seed(seed)
    cap_noise = np.random.normal(0, CAPACITY_NOISE_STD, size=len(dates))
    tp_noise = np.random.normal(0, CALLS_NOISE_STD, size=len(dates))
    capacity = _capacity_from_noise(dates, cap_noise)
    total_presented = _total_presented_from_noise(dates, tp_noise)
    return capacity, total_presented
