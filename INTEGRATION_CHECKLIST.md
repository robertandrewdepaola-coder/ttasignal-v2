# ✅ Market Structure Integration Checklist

## Quick Start (2 Steps)

### ☐ Step 1: Add Function to app.py

1. Open `app.py` in your editor
2. Find the `_run_factual_brief()` function (around line 1650)
3. Scroll down to the line **AFTER** that function ends
4. Open [`app_additions.py`](https://github.com/robertandrewdepaola-coder/ttasignal-v2/blob/main/app_additions.py)
5. Copy the `render_market_structure_section()` function
6. Paste it into `app.py` after `_run_factual_brief()`
7. Save the file

---

### ☐ Step 2: Update main() Function

1. Scroll to the **very end** of `app.py`
2. Look for a `main()` function

**If main() exists:**
- Replace it with the version from [`app_additions.py`](https://github.com/robertandrewdepaola-coder/ttasignal-v2/blob/main/app_additions.py)

**If main() doesn't exist:**
- Copy the entire `main()` function + `if __name__ == "__main__":` block from `app_additions.py`
- Paste at the absolute bottom of `app.py`

3. Save the file

---

## ✅ Done!

Run your app:
```bash
streamlit run app.py
```

You should see:
- 🏛️ **Market Structure Dashboard** (collapsible section above scanner)
- Click "▶️ Run Analysis" to generate the report
- Results cached for 30 minutes

---

## 📄 Reference

| File | Purpose |
|------|--------|
| [`INTEGRATE_MARKET_STRUCTURE.md`](INTEGRATE_MARKET_STRUCTURE.md) | **⭐ Start Here** - Detailed instructions with full code |
| [`app_additions.py`](app_additions.py) | Complete code to copy (function + main) |
| [`market_structure.py`](market_structure.py) | Core engine (already created) |
| `app.py` | **Edit this file** (your main app) |

---

## 🔧 Troubleshooting

**Dashboard not showing?**
➡️ Make sure you called `render_market_structure_section()` inside `main()`

**"module not found" error?**
➡️ Check that `market_structure.py` exists in your project root

**Analysis fails?**
➡️ Check Python console for data fetching errors (yfinance, network)

---

## 📊 What You Get

✅ Market Score (0-100)  
✅ Market Signal (Bullish/Cautious/Neutral/Bearish)  
✅ Market Regime (Risk-On/Transition/Defensive/Risk-Off)  
✅ 4-Factor Analysis (S&P 500, VIX, Dollar, Yields)  
✅ Sector Rotation Table (11 sectors ranked + classified)  
✅ 30-minute cache with manual refresh  

---

**Need help?** See [`INTEGRATE_MARKET_STRUCTURE.md`](INTEGRATE_MARKET_STRUCTURE.md) for detailed guide.
