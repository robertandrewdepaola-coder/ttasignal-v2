from __future__ import annotations

from typing import Dict, List

import signal_engine as se


def _dummy_ohlcv(rows: int = 80):
    # check_macd_mtf_zones only needs len(...) before normalize/calculate calls.
    return list(range(rows))


def _patch_zone_pipeline(monkeypatch, zones: List[Dict]):
    _zones = iter(zones)

    def _fake_calculate_macd(df, profile=None):
        return {"MACD": 0.0, "MACD_Signal": 0.0}

    def _fake_classify(df_macd, lookback_cross=5):
        return next(_zones)

    monkeypatch.setattr(se, "calculate_macd", _fake_calculate_macd)
    monkeypatch.setattr(se, "classify_macd_zone", _fake_classify)
    monkeypatch.setattr(se, "normalize_columns", lambda x: x)


def test_mtf_zone_allows_mature_weekly_monthly_when_daily_recent(monkeypatch):
    _patch_zone_pipeline(
        monkeypatch,
        zones=[
            {"zone": "extended", "recent_cross": True, "hist_pct": 0.88},  # daily
            {"zone": "extended", "recent_cross": False, "hist_pct": 0.75},  # weekly
            {"zone": "strong", "recent_cross": False, "hist_pct": 0.62},  # monthly
        ],
    )
    df = _dummy_ohlcv(90)
    out = se.check_macd_mtf_zones(df, df, df)
    assert out["buy_approved"] is True
    assert out["reject_reason"] is None


def test_mtf_zone_rejects_when_daily_not_fresh_entry_zone(monkeypatch):
    _patch_zone_pipeline(
        monkeypatch,
        zones=[
            {"zone": "strong", "recent_cross": False, "hist_pct": 0.72},  # daily
            {"zone": "extended", "recent_cross": False, "hist_pct": 0.70},  # weekly
            {"zone": "extended", "recent_cross": False, "hist_pct": 0.82},  # monthly
        ],
    )
    df = _dummy_ohlcv(90)
    out = se.check_macd_mtf_zones(df, df, df)
    assert out["buy_approved"] is False
    assert "Daily not in fresh entry zone" in str(out["reject_reason"])
