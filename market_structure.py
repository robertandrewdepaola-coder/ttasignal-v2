"""
Market Structure Analysis Engine
=================================

Multi-factor market timing and sector rotation analysis.

Factors analyzed:
1. S&P 500 momentum + trend filter (200-day MA)
2. VIX risk gauge
3. US Dollar strength (risk appetite)
4. Treasury yields (growth environment)
5. Sector rotation breadth and momentum

Version: 1.0.0 (2026-02-16)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

SECTOR_ETFS = {
    'XLE': 'Energy',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLRE': 'Real Estate',
    'XLV': 'Healthcare',
    'XLC': 'Communications',
    'XLK': 'Technology',
    'XLY': 'Consumer Discretionary',
    'XLF': 'Financials',
}

MARKET_INDICATORS = {
    'SPY': 'S&P 500',
    '^VIX': 'VIX',
    'UUP': 'US Dollar',
    'TLT': '20Y Treasury',
}

# Thresholds
VIX_ELEVATED = 20.0
VIX_HIGH = 25.0
TREND_MA_PERIOD = 200
MOMENTUM_SHORT = 5
MOMENTUM_MED = 20

# Rotation classification thresholds (% returns)
LEADING_THRESHOLD = 5.0  # 20d momentum > 5%
EMERGING_THRESHOLD = 2.0  # 20d momentum > 2%
FADING_THRESHOLD = 0.0    # 20d momentum > 0%
# Below 0% = Lagging

# Composite score weights
WEIGHTS = {
    'trend': 0.25,      # S&P above 200-day MA
    'momentum': 0.15,   # S&P 20-day momentum
    'vix': 0.15,        # VIX risk level
    'dollar': 0.15,     # Dollar strength
    'yields': 0.15,     # Treasury yields
    'rotation': 0.15,   # Sector rotation strength
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SectorSignal:
    """Individual sector analysis."""
    ticker: str
    name: str
    momentum_5d: float
    momentum_20d: float
    classification: str  # LEADING, EMERGING, FADING, LAGGING
    rank: int  # 1-11 ranking by 20d momentum


@dataclass
class MarketStructure:
    """Complete market structure snapshot."""
    timestamp: str
    
    # Trend
    spy_price: float
    spy_ma200: float
    trend_bullish: bool  # SPY > MA200
    spy_momentum_20d: float
    
    # Risk gauges
    vix: float
    vix_status: str  # LOW, ELEVATED, HIGH
    
    # Macro
    dollar_momentum_20d: float
    dollar_status: str  # STRENGTHENING, WEAKENING
    yields_momentum_20d: float
    yields_status: str  # RISING, FALLING
    
    # Rotation
    sectors: List[Dict]
    rotation_strength: int  # 0-100 score
    leading_count: int
    emerging_count: int
    fading_count: int
    lagging_count: int
    
    # Regime
    regime: str  # RISK_ON, RISK_OFF, TRANSITION, DEFENSIVE
    regime_confidence: int  # 0-100
    
    # Composite
    market_score: int  # 0-100 (composite of all factors)
    market_signal: str  # BULLISH, CAUTIOUS_BULLISH, NEUTRAL, CAUTIOUS_BEARISH, BEARISH
    recommendation: str  # Action recommendation
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# MARKET STRUCTURE ANALYZER
# =============================================================================

class MarketStructureAnalyzer:
    """Analyze market structure across multiple factors."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "market_structure_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load historical market structure snapshots."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[market_structure] Error loading history: {e}")
        return []
    
    def _save_snapshot(self, structure: MarketStructure):
        """Save market structure snapshot to history."""
        self.history.append(structure.to_dict())
        # Keep last 90 days only
        if len(self.history) > 90:
            self.history = self.history[-90:]
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"[market_structure] Error saving history: {e}")
    
    def fetch_data(self, lookback_days: int = 250) -> Dict[str, pd.DataFrame]:
        """Fetch market data for all indicators."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        data = {}
        tickers = list(MARKET_INDICATORS.keys()) + list(SECTOR_ETFS.keys())
        
        try:
            for ticker in tickers:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df
                else:
                    print(f"[market_structure] WARNING: No data for {ticker}")
        except Exception as e:
            print(f"[market_structure] Error fetching data: {e}")
        
        return data
    
    def calculate_momentum(self, df: pd.DataFrame, periods: int) -> float:
        """Calculate percentage momentum over N periods."""
        if len(df) < periods + 1:
            return 0.0
        current = df['Close'].iloc[-1]
        past = df['Close'].iloc[-(periods + 1)]
        return round((current - past) / past * 100, 2)
    
    def analyze_trend(self, spy_data: pd.DataFrame) -> Tuple[bool, float, float, float]:
        """Analyze S&P 500 trend vs 200-day MA."""
        if len(spy_data) < TREND_MA_PERIOD:
            return False, 0.0, 0.0, 0.0
        
        current_price = spy_data['Close'].iloc[-1]
        ma200 = spy_data['Close'].rolling(TREND_MA_PERIOD).mean().iloc[-1]
        momentum_20d = self.calculate_momentum(spy_data, MOMENTUM_MED)
        
        trend_bullish = current_price > ma200
        
        return trend_bullish, float(current_price), float(ma200), momentum_20d
    
    def analyze_vix(self, vix_data: pd.DataFrame) -> Tuple[float, str]:
        """Analyze VIX risk level."""
        if vix_data.empty:
            return 0.0, 'UNKNOWN'
        
        vix = float(vix_data['Close'].iloc[-1])
        
        if vix < VIX_ELEVATED:
            status = 'LOW'
        elif vix < VIX_HIGH:
            status = 'ELEVATED'
        else:
            status = 'HIGH'
        
        return vix, status
    
    def analyze_macro(self, dollar_data: pd.DataFrame, yields_data: pd.DataFrame) -> Dict:
        """Analyze Dollar and Treasury yields."""
        result = {
            'dollar_momentum_20d': 0.0,
            'dollar_status': 'UNKNOWN',
            'yields_momentum_20d': 0.0,
            'yields_status': 'UNKNOWN',
        }
        
        if not dollar_data.empty:
            dollar_mom = self.calculate_momentum(dollar_data, MOMENTUM_MED)
            result['dollar_momentum_20d'] = dollar_mom
            result['dollar_status'] = 'STRENGTHENING' if dollar_mom > 0 else 'WEAKENING'
        
        if not yields_data.empty:
            yields_mom = self.calculate_momentum(yields_data, MOMENTUM_MED)
            result['yields_momentum_20d'] = yields_mom
            result['yields_status'] = 'RISING' if yields_mom > 0 else 'FALLING'
        
        return result
    
    def analyze_sectors(self, data: Dict[str, pd.DataFrame]) -> Tuple[List[SectorSignal], Dict]:
        """Analyze sector rotation and classify sectors."""
        sector_signals = []
        
        for ticker, name in SECTOR_ETFS.items():
            if ticker not in data:
                continue
            
            df = data[ticker]
            mom_5d = self.calculate_momentum(df, MOMENTUM_SHORT)
            mom_20d = self.calculate_momentum(df, MOMENTUM_MED)
            
            # Classify
            if mom_20d >= LEADING_THRESHOLD:
                classification = 'LEADING'
            elif mom_20d >= EMERGING_THRESHOLD:
                classification = 'EMERGING'
            elif mom_20d >= FADING_THRESHOLD:
                classification = 'FADING'
            else:
                classification = 'LAGGING'
            
            sector_signals.append(SectorSignal(
                ticker=ticker,
                name=name,
                momentum_5d=mom_5d,
                momentum_20d=mom_20d,
                classification=classification,
                rank=0,  # Will be set after sorting
            ))
        
        # Rank by 20d momentum
        sector_signals.sort(key=lambda x: x.momentum_20d, reverse=True)
        for i, signal in enumerate(sector_signals, 1):
            signal.rank = i
        
        # Count classifications
        counts = {
            'leading': sum(1 for s in sector_signals if s.classification == 'LEADING'),
            'emerging': sum(1 for s in sector_signals if s.classification == 'EMERGING'),
            'fading': sum(1 for s in sector_signals if s.classification == 'FADING'),
            'lagging': sum(1 for s in sector_signals if s.classification == 'LAGGING'),
        }
        
        # Rotation strength score (0-100)
        # Strong rotation = many leading/emerging, few lagging
        rotation_strength = min(100, int(
            (counts['leading'] * 15) +
            (counts['emerging'] * 10) +
            (counts['fading'] * 5) +
            (11 - counts['lagging']) * 3
        ))
        
        counts['rotation_strength'] = rotation_strength
        
        return sector_signals, counts
    
    def classify_regime(self, trend_bullish: bool, vix: float, dollar_mom: float, 
                       yields_mom: float, leading_count: int, lagging_count: int) -> Tuple[str, int]:
        """Classify market regime."""
        
        # Score factors (0-100 each)
        scores = {
            'trend': 100 if trend_bullish else 0,
            'vix': 100 if vix < VIX_ELEVATED else (50 if vix < VIX_HIGH else 0),
            'dollar': 100 if dollar_mom < 0 else 0,  # Weak dollar = risk-on
            'yields': 100 if yields_mom < 0 else 50,  # Falling yields = growth-friendly
            'rotation': min(100, leading_count * 30),  # More leading sectors = risk-on
        }
        
        avg_score = sum(scores.values()) / len(scores)
        
        # Classify regime
        if avg_score >= 70:
            regime = 'RISK_ON'
        elif avg_score >= 50:
            regime = 'TRANSITION'
        elif leading_count >= 3 and lagging_count <= 2:
            regime = 'DEFENSIVE'  # Defensive rotation but not full risk-off
        else:
            regime = 'RISK_OFF'
        
        confidence = int(avg_score)
        
        return regime, confidence
    
    def calculate_market_score(self, trend_bullish: bool, spy_momentum: float, 
                              vix: float, dollar_mom: float, yields_mom: float,
                              rotation_strength: int) -> int:
        """Calculate composite market score (0-100)."""
        
        # Individual component scores (0-100)
        trend_score = 100 if trend_bullish else 0
        momentum_score = max(0, min(100, 50 + spy_momentum * 5))  # Centered at 50
        vix_score = 100 if vix < VIX_ELEVATED else (50 if vix < VIX_HIGH else 0)
        dollar_score = max(0, min(100, 50 - dollar_mom * 5))  # Weak dollar = higher score
        yields_score = max(0, min(100, 50 - yields_mom * 5))  # Falling yields = higher score
        
        # Weighted composite
        composite = (
            trend_score * WEIGHTS['trend'] +
            momentum_score * WEIGHTS['momentum'] +
            vix_score * WEIGHTS['vix'] +
            dollar_score * WEIGHTS['dollar'] +
            yields_score * WEIGHTS['yields'] +
            rotation_strength * WEIGHTS['rotation']
        )
        
        return int(composite)
    
    def get_market_signal(self, market_score: int, trend_bullish: bool) -> Tuple[str, str]:
        """Convert market score to actionable signal."""
        
        # Require trend confirmation for bullish signals
        if market_score >= 70 and trend_bullish:
            signal = 'BULLISH'
            recommendation = '✅ AGGRESSIVE: Add positions in leading sectors. Full risk-on.'
        elif market_score >= 55:
            if trend_bullish:
                signal = 'CAUTIOUS_BULLISH'
                recommendation = '🟢 SELECTIVE: Trade leading sectors only. Normal position sizing.'
            else:
                signal = 'NEUTRAL'
                recommendation = '🟡 WAIT: Trend not confirmed. Watch for S&P to clear 200-day MA.'
        elif market_score >= 40:
            signal = 'NEUTRAL'
            recommendation = '🟡 DEFENSIVE: Reduce exposure. Focus on defensive sectors or cash.'
        elif market_score >= 25:
            signal = 'CAUTIOUS_BEARISH'
            recommendation = '🟠 CAUTION: High risk environment. Tighten stops, avoid new longs.'
        else:
            signal = 'BEARISH'
            recommendation = '🔴 RISK-OFF: Exit positions. Cash or short-only strategies.'
        
        return signal, recommendation
    
    def analyze(self) -> MarketStructure:
        """Run complete market structure analysis."""
        print("[market_structure] Fetching data...")
        data = self.fetch_data()
        
        if not data:
            raise ValueError("No market data available")
        
        # Trend analysis
        trend_bullish, spy_price, spy_ma200, spy_momentum = self.analyze_trend(data.get('SPY'))
        
        # VIX analysis
        vix, vix_status = self.analyze_vix(data.get('^VIX'))
        
        # Macro analysis
        macro = self.analyze_macro(data.get('UUP'), data.get('TLT'))
        
        # Sector rotation
        sector_signals, counts = self.analyze_sectors(data)
        
        # Regime classification
        regime, regime_confidence = self.classify_regime(
            trend_bullish, vix, macro['dollar_momentum_20d'], 
            macro['yields_momentum_20d'], counts['leading'], counts['lagging']
        )
        
        # Market score
        market_score = self.calculate_market_score(
            trend_bullish, spy_momentum, vix,
            macro['dollar_momentum_20d'], macro['yields_momentum_20d'],
            counts['rotation_strength']
        )
        
        # Signal
        market_signal, recommendation = self.get_market_signal(market_score, trend_bullish)
        
        # Build structure
        structure = MarketStructure(
            timestamp=datetime.now().isoformat(),
            spy_price=spy_price,
            spy_ma200=spy_ma200,
            trend_bullish=trend_bullish,
            spy_momentum_20d=spy_momentum,
            vix=vix,
            vix_status=vix_status,
            dollar_momentum_20d=macro['dollar_momentum_20d'],
            dollar_status=macro['dollar_status'],
            yields_momentum_20d=macro['yields_momentum_20d'],
            yields_status=macro['yields_status'],
            sectors=[asdict(s) for s in sector_signals],
            rotation_strength=counts['rotation_strength'],
            leading_count=counts['leading'],
            emerging_count=counts['emerging'],
            fading_count=counts['fading'],
            lagging_count=counts['lagging'],
            regime=regime,
            regime_confidence=regime_confidence,
            market_score=market_score,
            market_signal=market_signal,
            recommendation=recommendation,
        )
        
        # Save snapshot
        self._save_snapshot(structure)
        
        return structure
    
    def format_report(self, structure: MarketStructure) -> str:
        """Format market structure as readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("MARKET STRUCTURE ANALYSIS")
        lines.append(f"Timestamp: {structure.timestamp}")
        lines.append("=" * 70)
        lines.append("")
        
        # Market Score
        lines.append(f"📊 MARKET SCORE: {structure.market_score}/100 — {structure.market_signal}")
        lines.append(f"   {structure.recommendation}")
        lines.append("")
        
        # Regime
        lines.append(f"🌍 REGIME: {structure.regime} (confidence: {structure.regime_confidence}/100)")
        lines.append("")
        
        # Trend
        trend_emoji = '🟢' if structure.trend_bullish else '🔴'
        lines.append(f"{trend_emoji} S&P 500: ${structure.spy_price:.2f} vs MA200 ${structure.spy_ma200:.2f}")
        lines.append(f"   Trend: {'Bullish' if structure.trend_bullish else 'Bearish'} | 20d momentum: {structure.spy_momentum_20d:+.1f}%")
        lines.append("")
        
        # Risk Gauges
        vix_emoji = '🟢' if structure.vix < VIX_ELEVATED else ('🟠' if structure.vix < VIX_HIGH else '🔴')
        lines.append(f"{vix_emoji} VIX: {structure.vix:.1f} — {structure.vix_status}")
        
        dollar_emoji = '🟢' if structure.dollar_momentum_20d < 0 else '🔴'
        lines.append(f"{dollar_emoji} US Dollar: {structure.dollar_momentum_20d:+.1f}% — {structure.dollar_status}")
        
        yields_emoji = '🟢' if structure.yields_momentum_20d < 0 else '🔴'
        lines.append(f"{yields_emoji} Treasury Yields (TLT): {structure.yields_momentum_20d:+.1f}% — {structure.yields_status}")
        lines.append("")
        
        # Rotation
        lines.append(f"📈 SECTOR ROTATION (Strength: {structure.rotation_strength}/100)")
        lines.append(f"   Leading: {structure.leading_count} | Emerging: {structure.emerging_count} | "
                    f"Fading: {structure.fading_count} | Lagging: {structure.lagging_count}")
        lines.append("")
        
        # Sectors by classification
        for classification in ['LEADING', 'EMERGING', 'FADING', 'LAGGING']:
            emoji_map = {
                'LEADING': '🟢',
                'EMERGING': '🔵',
                'FADING': '🟡',
                'LAGGING': '🔴',
            }
            sectors_in_class = [s for s in structure.sectors if s['classification'] == classification]
            if sectors_in_class:
                lines.append(f"{emoji_map[classification]} {classification}:")
                for s in sectors_in_class:
                    lines.append(f"   {s['ticker']} {s['name']:20s} 5d:{s['momentum_5d']:+5.1f}% 20d:{s['momentum_20d']:+5.1f}%")
                lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    analyzer = MarketStructureAnalyzer()
    structure = analyzer.analyze()
    print(analyzer.format_report(structure))
