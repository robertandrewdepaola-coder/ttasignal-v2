import importlib
import sys

import numpy as np

# Some lightweight tests replace pandas in sys.modules with a stub.
if "pandas" in sys.modules and not hasattr(sys.modules["pandas"], "Series"):
    sys.modules.pop("pandas", None)
pd = importlib.import_module("pandas")

import signal_engine as _signal_engine

_signal_engine = importlib.reload(_signal_engine)
detect_vcp = _signal_engine.detect_vcp


def _build_ohlcv_with_segments(segment_ranges, total_len=80, vol_right=500_000, vol_left=1_000_000):
    close = np.linspace(100.0, 140.0, total_len)
    seg_count = len(segment_ranges)
    seg_len = total_len // seg_count

    high = np.zeros(total_len, dtype=float)
    low = np.zeros(total_len, dtype=float)
    volume = np.full(total_len, vol_left, dtype=float)

    for i, rng in enumerate(segment_ranges):
        start = i * seg_len
        end = (i + 1) * seg_len
        amp = float(rng) / 2.0
        high[start:end] = close[start:end] + amp
        low[start:end] = close[start:end] - amp

    volume[-seg_len:] = vol_right
    open_ = close * 0.998

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_detect_vcp_positive_case():
    # Left-to-right contraction + right-side volume contraction + uptrend.
    df = _build_ohlcv_with_segments([20, 12, 8, 4], total_len=80, vol_right=450_000, vol_left=1_000_000)
    out = detect_vcp(df, lookback=80, n_segments=4, min_contractions=2, volume_contraction_threshold=0.75)

    assert out["price_contracting"] is True
    assert out["volume_contracting"] is True
    assert out["in_uptrend"] is True
    assert out["vcp_detected"] is True
    assert out["vcp_score"] > 0
    assert out["pivot_price"] is not None


def test_detect_vcp_negative_case_no_contraction():
    # Flat ranges and no volume contraction should fail VCP.
    df = _build_ohlcv_with_segments([10, 10, 10, 10], total_len=80, vol_right=1_000_000, vol_left=1_000_000)
    out = detect_vcp(df, lookback=80, n_segments=4, min_contractions=2, volume_contraction_threshold=0.75)

    assert out["price_contracting"] is False
    assert out["volume_contracting"] is False
    assert out["vcp_detected"] is False
