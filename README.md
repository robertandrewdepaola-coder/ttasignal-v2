# TTA Signal v2 — Technical Trading Assistant

AI-assisted swing trading scanner using MACD + Awesome Oscillator confluence signals.

## Architecture

| Module | Role |
|--------|------|
| `signal_engine.py` | All indicator calculations, Weinstein stages, overhead resistance, volume analysis |
| `data_fetcher.py` | yfinance data with session caching, options, insider, institutional data |
| `scanner_engine.py` | Quality scoring (mini-backtest), signal classification, recommendations |
| `ai_analysis.py` | AI-enhanced trade intelligence via Gemini/OpenAI |
| `chart_engine.py` | Plotly candlestick charts with indicator overlays |
| `journal_manager.py` | Trade lifecycle: watchlist → entry → management → exit → P&L |
| `app.py` | Streamlit UI (thin layer) |

## Setup

1. Clone this repo
2. `pip install -r requirements.txt`
3. Add your Gemini API key to `.streamlit/secrets.toml` (see template)
4. `streamlit run app.py`

## Streamlit Cloud

Connect this repo at [share.streamlit.io](https://share.streamlit.io). Set `GEMINI_API_KEY` in Settings → Secrets.
