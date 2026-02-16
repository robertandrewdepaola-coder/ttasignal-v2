# 🏛️ Market Structure Dashboard - Complete Integration Package

## 🎯 Overview

This package integrates a **real-time Market Structure Dashboard** into your TTA v2 trading app. It provides multi-factor market timing analysis and sector rotation intelligence to enhance trading decisions.

---

## ⚡ Quick Start (2 Steps)

### 📚 Read This First: [`INTEGRATION_CHECKLIST.md`](INTEGRATION_CHECKLIST.md)

**Just 2 copy-paste steps:**
1. Add `render_market_structure_section()` function to `app.py`
2. Update `main()` function at the end of `app.py`

**Total time:** 5 minutes

---

## 📊 What You Get

### Market Structure Dashboard Features

✅ **Market Score** (0-100) - Overall market health  
✅ **Market Signal** - Bullish/Cautious/Neutral/Bearish  
✅ **Market Regime** - Risk-On/Transition/Defensive/Risk-Off with confidence %  
✅ **4-Factor Analysis**:
  - S&P 500 trend + momentum (20d)
  - VIX status (LOW/ELEVATED/HIGH)
  - US Dollar momentum (bullish/bearish for stocks)
  - TLT Yields trend (cost of money)

✅ **Sector Rotation Table** (11 sectors):
  - Rank by relative strength
  - 5-day and 20-day momentum vs SPY
  - Phase classification (Leading/Emerging/Fading/Lagging)
  - ETF tickers for each sector

✅ **Smart Caching** - 30-minute cache with manual refresh button

---

## 📂 Files in This Package

| File | Purpose | Start Here? |
|------|---------|-------------|
| **[`INTEGRATION_CHECKLIST.md`](INTEGRATION_CHECKLIST.md)** | ⭐ **Quick 2-step guide** | 👉 **YES** |
| **[`INTEGRATE_MARKET_STRUCTURE.md`](INTEGRATE_MARKET_STRUCTURE.md)** | Detailed instructions with full code | If you need details |
| **[`app_additions.py`](app_additions.py)** | Complete code to copy | Copy from here |
| **[`market_structure.py`](market_structure.py)** | Core analysis engine | Already created |
| **[`MARKET_STRUCTURE_INTEGRATION.md`](MARKET_STRUCTURE_INTEGRATION.md)** | Original detailed guide | Reference |
| `app.py` | Your main app file | **Edit this** |

---

## 🚀 Installation

### Prerequisites

All dependencies already in your `requirements.txt`:
- `yfinance` (market data)
- `pandas` (data processing)
- `streamlit` (UI)

### Integration Steps

1. **Read the checklist:**  
   Open [`INTEGRATION_CHECKLIST.md`](INTEGRATION_CHECKLIST.md)

2. **Copy code:**  
   Get code from [`app_additions.py`](app_additions.py)

3. **Edit app.py:**  
   - Add `render_market_structure_section()` after `_run_factual_brief()`
   - Update `main()` at the end of the file

4. **Test:**
   ```bash
   streamlit run app.py
   ```

5. **Use:**  
   Look for "🏛️ Market Structure Dashboard" section above the scanner table

---

## 🔍 Usage

### First Time

1. Open your TTA v2 app
2. Look for the **"🏛️ Market Structure Dashboard"** collapsible section (above scanner)
3. Click **"▶️ Run Analysis"**
4. Dashboard generates in ~3-5 seconds
5. Results cached for 30 minutes

### Daily Workflow

**Morning (Pre-Market):**
- Open app → Expand Market Structure Dashboard
- Check Market Score + Regime
- Review sector rotation (Leading vs Lagging)
- Plan trades based on sector strength

**During Market:**
- Dashboard stays collapsed (doesn't interfere with scanning)
- Expand when you need macro context for a trade decision

**After 30 Minutes:**
- Click "🔄 Refresh" to update data
- Or leave cached until next session (still useful)

---

## 💡 Trading Application

### Interpreting the Dashboard

**Market Score:**
- 70-100: Strong bullish environment → Full position sizing
- 55-69: Neutral-bullish → Selective entries
- 40-54: Caution zone → Reduce size, tighten stops
- 0-39: Defensive mode → Cash heavy, shorts only

**Market Regime:**
- **Risk-On** → Trade aggressively, favor growth/tech
- **Transition** → Be selective, quick profits
- **Defensive** → Favor defensive sectors (Utilities, Staples)
- **Risk-Off** → Cash/shorts, avoid new longs

**Sector Rotation:**
- **Leading** (green) → Primary trade candidates
- **Emerging** (blue) → Watch for momentum confirmation
- **Fading** (yellow) → Tighten stops, take profits
- **Lagging** (red) → Avoid new entries

---

## ⚙️ Technical Details

### Data Sources

- **S&P 500 (SPY):** Price, 200-day MA, 20-day momentum
- **VIX (^VIX):** Current level, 20-day average
- **US Dollar (UUP):** 20-day momentum
- **TLT (Bonds):** 20-day momentum (inverse of yields)
- **11 Sector ETFs:** XLK, XLF, XLV, XLY, XLP, XLE, XLU, XLI, XLRE, XLB, XLC

### Scoring Algorithm

**Market Score** (0-100):
- SPY > 200 MA: +25 points
- VIX < 20: +25 points
- Dollar declining: +25 points (bullish for stocks)
- Yields declining: +25 points (easy money)

**Sector Classification:**
- **Leading:** 5d > +1% vs SPY AND 20d > +2% vs SPY
- **Emerging:** 5d > +1% OR 20d > +2% (but not both)
- **Fading:** 5d < -1% vs SPY OR 20d < -2% vs SPY
- **Lagging:** Both 5d and 20d underperforming

### Caching Strategy

- **Cache Duration:** 30 minutes (1800 seconds)
- **Storage:** Streamlit session state
- **Refresh:** Manual via button (avoids API spam)
- **Why 30 min?** Market structure changes slowly; frequent updates unnecessary

---

## 🔧 Troubleshooting

### Common Issues

**Error: "market_structure module not found"**
➡️ Check that `market_structure.py` exists in your project root

**Error: "MarketStructureAnalyzer not found"**
➡️ Verify `market_structure.py` has the `MarketStructureAnalyzer` class

**Dashboard section not showing**
➡️ Make sure you added `render_market_structure_section()` call in `main()`

**Analysis button does nothing**
➡️ Check Python console for errors (likely network/yfinance issue)

**Data looks stale**
➡️ Click the "🔄 Refresh" button to force a new fetch

### Debug Mode

Add this to your `market_structure.py` to see what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Then check console for:
- Data fetch status
- Calculation results
- Cache hits/misses

---

## 📝 Changelog

### v1.0.0 (2026-02-16)
- ✅ Initial release
- ✅ 4-factor market analysis (SPY, VIX, Dollar, Yields)
- ✅ 11-sector rotation tracking
- ✅ Phase classification (Leading/Emerging/Fading/Lagging)
- ✅ Market regime detection (Risk-On/Transition/Defensive/Risk-Off)
- ✅ 30-minute smart caching
- ✅ Integration with TTA v2 scanner

---

## ❓ FAQ

**Q: Will this slow down my app?**  
A: No. Dashboard only runs when you click "Run Analysis". Results cached for 30 minutes.

**Q: Do I need extra API keys?**  
A: No. Uses yfinance (free, no key needed).

**Q: Can I customize the sectors?**  
A: Yes! Edit the `SECTOR_ETFS` dict in `market_structure.py`.

**Q: How often should I refresh?**  
A: Pre-market (once), mid-day (once), end-of-day (once). Market structure doesn't change intraday.

**Q: Does it work with existing features?**  
A: Yes! Fully compatible with scanner, AI Intel, chart view, etc.

---

## 👏 Credits

**Market Structure Dashboard** by TTA v2 Team

Based on:
- Juan's Market Filter methodology (5-factor analysis)
- Sector rotation tracking (RRG-inspired)
- Multi-timeframe momentum confirmation

---

## 💬 Support

Need help?
1. Check [`INTEGRATION_CHECKLIST.md`](INTEGRATION_CHECKLIST.md) first
2. Review [`INTEGRATE_MARKET_STRUCTURE.md`](INTEGRATE_MARKET_STRUCTURE.md) for details
3. Check Python console for error messages
4. Verify all files exist in your project root

---

## ⭐ Start Now

**Ready to integrate?**

👉 **[Open INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md)** 👈

Just 2 copy-paste steps. 5 minutes total.

Happy trading! 📈
