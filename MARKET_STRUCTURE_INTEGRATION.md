# Market Structure Dashboard Integration Guide

## Step 1: Add the Function to app.py

**Location:** After `_run_factual_brief()` function (around line 1650)

```python
def render_market_structure_section():
    """Market Structure Dashboard — multi-factor timing analysis (standalone section)."""
    from market_structure import MarketStructureAnalyzer
    
    # Cache analysis for 30 minutes
    cache_key = 'market_structure_data'
    cache_ts_key = 'market_structure_ts'
    now = datetime.now().timestamp()
    cached_ts = st.session_state.get(cache_ts_key, 0)
    stale = (now - cached_ts) > 1800  # 30 min TTL

    with st.expander("🏛️ Market Structure Dashboard", expanded=False):
        # Show run button if no cache or stale
        if cache_key not in st.session_state or stale:
            st.info("📊 **Real-time market analysis** — Multi-factor timing + sector rotation")
            if st.button("▶️ Run Analysis", type="primary", key="ms_run"):
                with st.spinner("Analyzing market structure..."):
                    try:
                        analyzer = MarketStructureAnalyzer()
                        structure = analyzer.analyze()
                        st.session_state[cache_key] = structure.to_dict()
                        st.session_state[cache_ts_key] = now
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            return

        structure = st.session_state.get(cache_key)
        if not structure:
            return

        # === TOP ROW: Score + Signal + Regime ===
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            score = structure['market_score']
            signal = structure['market_signal']
            if score >= 70:
                st.success(f"**{score}/100**")
            elif score >= 55:
                st.info(f"**{score}/100**")
            elif score >= 40:
                st.warning(f"**{score}/100**")
            else:
                st.error(f"**{score}/100**")
            st.caption(f"**{signal}**")
        
        with col2:
            regime = structure['regime']
            conf = structure['regime_confidence']
            regime_emoji = {
                'RISK_ON': '🟢',
                'TRANSITION': '🟡',
                'DEFENSIVE': '🟠',
                'RISK_OFF': '🔴',
            }.get(regime, '⚪')
            st.metric("Market Regime", f"{regime_emoji} {regime}", f"{conf}% confidence")
        
        with col3:
            st.info(structure['recommendation'])

        st.divider()

        # === FACTOR GRID ===
        f1, f2, f3, f4 = st.columns(4)
        
        with f1:
            trend_icon = "✅" if structure['trend_bullish'] else "❌"
            mom = structure['spy_momentum_20d']
            st.metric("S&P 500", f"{trend_icon} ${structure['spy_price']:.2f}", 
                     delta=f"{mom:+.1f}% (20d)")
            st.caption(f"vs 200 MA: ${structure['spy_ma200']:.2f}")
        
        with f2:
            vix_icon = {'LOW': '🟢', 'ELEVATED': '🟠', 'HIGH': '🔴'}.get(structure['vix_status'], '⚪')
            st.metric("VIX", f"{vix_icon} {structure['vix']:.1f}", structure['vix_status'])
        
        with f3:
            dollar_icon = "🟢" if structure['dollar_momentum_20d'] < 0 else "🔴"
            st.metric("US Dollar", f"{dollar_icon} {structure['dollar_momentum_20d']:+.1f}%", 
                     structure['dollar_status'])
        
        with f4:
            yield_icon = "🟢" if structure['yields_momentum_20d'] < 0 else "🔴"
            st.metric("TLT Yields", f"{yield_icon} {structure['yields_momentum_20d']:+.1f}%",
                     structure['yields_status'])

        # === SECTOR ROTATION TABLE ===
        st.markdown("### 📊 Sector Rotation")
        
        rot_cols = st.columns([1, 1, 1, 1])
        rot_cols[0].metric("Leading", structure['leading_count'], "🟢")
        rot_cols[1].metric("Emerging", structure['emerging_count'], "🔵")
        rot_cols[2].metric("Fading", structure['fading_count'], "🟡")
        rot_cols[3].metric("Lagging", structure['lagging_count'], "🔴")
        
        st.caption(f"**Rotation Strength: {structure['rotation_strength']}/100**")

        # Sector table
        sector_data = []
        for s in structure['sectors']:
            sector_data.append({
                'Rank': s['rank'],
                'Sector': s['name'],
                'ETF': s['ticker'],
                '5d %': f"{s['momentum_5d']:+.1f}",
                '20d %': f"{s['momentum_20d']:+.1f}",
                'Phase': s['classification'],
            })
        
        st.dataframe(
            pd.DataFrame(sector_data),
            hide_index=True,
            use_container_width=True,
        )

        # Refresh button
        rcol1, rcol2 = st.columns([3, 1])
        with rcol2:
            if st.button("🔄 Refresh", key="ms_refresh"):
                st.session_state.pop(cache_key, None)
                st.session_state.pop(cache_ts_key, None)
                st.rerun()
        with rcol1:
            age = (now - st.session_state.get(cache_ts_key, now)) / 60
            st.caption(f"Last updated: {age:.0f} minutes ago")
```

---

## Step 2: Update main() Function

**Location:** At the very end of app.py

**Find this:**
```python
# At the end of the file, look for:
if __name__ == "__main__":
    main()
```

**Replace the main() function with:**
```python
def main():
    """Main entry point with integrated market structure dashboard."""
    render_sidebar()
    
    # === MARKET STRUCTURE DASHBOARD (standalone section) ===
    render_market_structure_section()
    
    st.divider()
    
    render_scanner_table()
    
    # Detail view
    if st.session_state.get('selected_ticker'):
        render_detail_view()


if __name__ == "__main__":
    main()
```

---

## What This Adds

✅ **Collapsible section above scanner table**  
✅ **Market score (0-100) + signal**  
✅ **Market regime (Risk-On/Transition/Defensive/Risk-Off)**  
✅ **4 factor grid: S&P 500, VIX, Dollar, Yields**  
✅ **Sector rotation table with 11 sectors ranked**  
✅ **30-minute cache with manual refresh button**  

---

## Quick Copy-Paste

**If main() doesn't exist yet**, add this at the very end of app.py:

```python
# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point with integrated market structure dashboard."""
    render_sidebar()
    
    # === MARKET STRUCTURE DASHBOARD ===
    render_market_structure_section()
    
    st.divider()
    
    render_scanner_table()
    
    # Detail view
    if st.session_state.get('selected_ticker'):
        render_detail_view()


if __name__ == "__main__":
    main()
```

---

## Testing

1. Restart your Streamlit app
2. Look for "🏛️ Market Structure Dashboard" section above the scanner table
3. Click "▶️ Run Analysis"
4. Review the market score, regime, and sector rotation

---

## Need Help?

If you encounter errors:
1. Check that `market_structure.py` exists and has `MarketStructureAnalyzer` class
2. Verify all imports at the top of app.py include `from market_structure import MarketStructureAnalyzer`
3. Check the Python console for stack traces
