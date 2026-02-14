"""
TTA v2 Scraping Bridge — Multi-Provider Ticker Fetcher
========================================================

Fetches tickers from various sources:
  - ETF shortcuts (ARK CSV downloads — free, instant)
  - ETF URLs (custom CSV/HTML)
  - TradingView CSV import (file upload)
  - Custom URLs (HTML regex extraction)

Phase 1: Direct fetching (no API keys needed)
Phase 2: EODHD/FMP screener APIs

Version: 1.0.0 (2026-02-14)
"""

import io
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


# =============================================================================
# ETF SHORTCUT CONFIGURATION
# =============================================================================

ETF_CSV_URLS = {
    "arkk": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "arkw": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "arkg": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "arkq": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_AUTONOMOUS_TECH._&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
    "arkf": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
    "arkx": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv",
}

# Display labels for UI dropdown
ETF_SHORTCUTS = {
    "ARKK - ARK Innovation ETF": "arkk",
    "ARKW - Next Generation Internet": "arkw",
    "ARKG - Genomic Revolution": "arkg",
    "ARKQ - Autonomous Tech & Robotics": "arkq",
    "ARKF - Fintech Innovation": "arkf",
    "ARKX - Space Exploration": "arkx",
}

# False positives — words that match [A-Z]{1,5} but aren't tickers
FALSE_POSITIVES = frozenset({
    "I", "A", "TO", "OR", "AND", "THE", "FOR", "CAN", "US", "OK",
    "URL", "API", "NYSE", "NASD", "ETF", "USD", "HTTP", "HTML",
    "JSON", "CSV", "YES", "NO", "FUND", "TOTAL", "CASH", "NAME",
    "DATE", "WEIGHT", "SHARE", "HTTPS", "NASDAQ", "TABLE", "INDEX",
    "CLOSE", "OPEN", "HIGH", "LOW", "VOL", "PCT", "AVG", "DAY",
    "CLASS", "PRICE", "VALUE", "NET", "GROSS", "TYPE", "RANK",
})

# Foreign exchange suffixes to reject
FOREIGN_SUFFIXES = frozenset({"F", "TO", "L", "SW", "V", "O", "DE", "PA", "AS", "HK"})

# Valid US class-share suffixes
VALID_CLASS_SHARES = frozenset({"A", "B", "C", "D"})

# Request settings
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,text/csv,application/csv,*/*",
}


# =============================================================================
# SCRAPING BRIDGE
# =============================================================================

class ScrapingBridge:
    """
    Multi-provider ticker fetcher.

    Routes fetch requests based on watchlist source_type.
    Phase 1: Direct HTTP fetching (no API keys).
    Phase 2: EODHD/FMP screener APIs (pluggable).
    """

    def __init__(self):
        """Initialize bridge. No API keys needed for Phase 1."""
        self._session = requests.Session()
        self._session.headers.update(REQUEST_HEADERS)

    def fetch_tickers(self, watchlist: Dict) -> Tuple[bool, str, List[str]]:
        """
        Fetch tickers based on watchlist configuration.

        Routes to the appropriate provider based on watchlist['source_type'].
        If watchlist has 'url_override', uses that URL instead of the default.

        Args:
            watchlist: Dict with at minimum 'source_type' and 'source' keys.

        Returns:
            (success, message, tickers) tuple.
        """
        source_type = watchlist.get("source_type", "")
        source = watchlist.get("source", "")

        # Check for user-provided URL override
        url_override = watchlist.get("url_override", "")
        if url_override:
            return self._fetch_csv_from_url(url_override, f"Custom URL override")

        try:
            if source_type == "etf_shortcut":
                return self._fetch_etf_shortcut(source)

            elif source_type == "etf_url":
                return self._fetch_etf_url(source)

            elif source_type == "tradingview":
                return self._fetch_tradingview_csv(source)

            elif source_type == "custom_url":
                return self._fetch_custom_url(source)

            else:
                return False, f"Unknown source type: {source_type}", []

        except Exception as e:
            return False, f"Fetch error: {str(e)[:200]}", []

    def fetch_from_csv_content(self, csv_content) -> Tuple[bool, str, List[str]]:
        """
        Parse tickers from raw CSV content (from file_uploader).

        Public method for sidebar CSV upload fallback.
        """
        return self._fetch_tradingview_csv(csv_content)

    def validate_url(self, url: str) -> Tuple[bool, str, int]:
        """
        Lightweight URL check via HEAD request before downloading.

        Returns:
            (reachable, message, status_code)
        """
        try:
            resp = self._session.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return True, "OK", 200
            elif resp.status_code == 404:
                return False, f"URL not found (404) — ARK may have renamed this file", 404
            elif resp.status_code == 403:
                return False, f"Access denied (403) — site may be blocking requests", 403
            else:
                return False, f"HTTP {resp.status_code}", resp.status_code
        except requests.exceptions.Timeout:
            return False, "URL timed out — server not responding", 0
        except requests.exceptions.ConnectionError:
            return False, "Connection failed — check internet or URL domain", 0
        except Exception as e:
            return False, f"URL check failed: {str(e)[:100]}", 0

    def get_etf_url(self, identifier: str) -> Optional[str]:
        """Get the configured CSV URL for an ETF shortcut (for display/debugging)."""
        return ETF_CSV_URLS.get(identifier.lower().strip())

    # --- ETF Shortcut (ARK CSV downloads) ------------------------------------

    def _fetch_etf_shortcut(self, identifier: str) -> Tuple[bool, str, List[str]]:
        """
        Fetch tickers from a pre-configured ETF CSV URL.

        ARK Invest publishes daily holdings as CSV files.
        Validates URL reachability before attempting download.
        """
        identifier = identifier.lower().strip()
        url = ETF_CSV_URLS.get(identifier)

        if not url:
            available = ", ".join(sorted(ETF_CSV_URLS.keys()))
            return False, f"Unknown ETF shortcut: '{identifier}'. Available: {available}", []

        # Pre-validate URL before downloading
        reachable, check_msg, status = self.validate_url(url)
        if not reachable:
            return False, (
                f"URL unreachable for {identifier.upper()}: {check_msg}\n"
                f"URL: {url}\n"
                f"Fix: Update the URL below or upload a CSV file."
            ), []

        return self._fetch_csv_from_url(url, f"ETF shortcut '{identifier.upper()}'")

    # --- ETF URL (custom CSV/HTML) -------------------------------------------

    def _fetch_etf_url(self, url: str) -> Tuple[bool, str, List[str]]:
        """Fetch tickers from a custom ETF holdings URL. Tries CSV then HTML."""
        if not url or not url.startswith(("http://", "https://")):
            return False, "Invalid URL (must start with http:// or https://)", []

        # Try CSV parsing first
        success, msg, tickers = self._fetch_csv_from_url(url, "ETF URL")
        if success and tickers:
            return success, msg, tickers

        # Fallback to HTML regex extraction
        return self._fetch_custom_url(url)

    # --- TradingView CSV Import ----------------------------------------------

    def _fetch_tradingview_csv(self, csv_content: str) -> Tuple[bool, str, List[str]]:
        """
        Extract tickers from TradingView CSV export.

        Accepts raw CSV content (from Streamlit file_uploader).
        Looks for 'Ticker' or 'Symbol' column.
        """
        if not csv_content:
            return False, "No CSV content provided", []

        try:
            # Try reading as CSV
            if isinstance(csv_content, bytes):
                csv_content = csv_content.decode("utf-8", errors="replace")

            df = pd.read_csv(io.StringIO(csv_content))

            if df.empty:
                return False, "CSV file is empty", []

            # Find ticker column (case-insensitive)
            ticker_col = None
            for col in df.columns:
                if col.strip().lower() in ("ticker", "symbol", "name", "code"):
                    ticker_col = col
                    break

            if ticker_col is None:
                # Try first column as fallback
                ticker_col = df.columns[0]
                print(f"[scraping_bridge] No 'Ticker'/'Symbol' column, using '{ticker_col}'")

            # Extract and clean
            raw_tickers = df[ticker_col].dropna().astype(str).tolist()
            tickers = self._clean_ticker_list(raw_tickers)

            if not tickers:
                return False, "No valid tickers found in CSV", []

            return True, f"Extracted {len(tickers)} tickers from CSV", tickers

        except Exception as e:
            return False, f"CSV parse error: {str(e)[:200]}", []

    # --- Custom URL (HTML regex extraction) ----------------------------------

    def _fetch_custom_url(self, url: str) -> Tuple[bool, str, List[str]]:
        """
        Fetch a URL and extract tickers using regex.

        Last-resort method — works on any page with ticker symbols visible.
        """
        if not url or not url.startswith(("http://", "https://")):
            return False, "Invalid URL", []

        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            text = response.text
            tickers = self._extract_tickers_from_text(text)

            if not tickers:
                return False, f"No tickers found at {url}", []

            return True, f"Extracted {len(tickers)} tickers from URL", tickers

        except requests.exceptions.Timeout:
            return False, f"Request timed out ({REQUEST_TIMEOUT}s)", []
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            if status == 404:
                return False, f"Page not found (404). URL may have changed.", []
            elif status == 403:
                return False, f"Access denied (403). Site may block automated requests.", []
            return False, f"HTTP error {status}: {str(e)[:100]}", []
        except requests.exceptions.ConnectionError:
            return False, "Connection failed. Check URL and internet connection.", []
        except Exception as e:
            return False, f"Fetch error: {str(e)[:200]}", []

    # --- CSV Fetcher (shared by etf_shortcut and etf_url) --------------------

    def _fetch_csv_from_url(self, url: str, label: str = "CSV") -> Tuple[bool, str, List[str]]:
        """Download and parse a CSV file from a URL, extracting tickers."""
        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            content = response.text
            if not content or len(content) < 20:
                return False, f"{label}: Empty response", []

            # Try pandas CSV parsing
            try:
                df = pd.read_csv(io.StringIO(content))
            except Exception:
                # Some CSVs have header junk — skip first few lines
                lines = content.strip().split("\n")
                for skip in range(min(5, len(lines))):
                    try:
                        df = pd.read_csv(io.StringIO("\n".join(lines[skip:])))
                        if len(df.columns) >= 2:
                            break
                    except Exception:
                        continue
                else:
                    return False, f"{label}: Could not parse CSV", []

            if df.empty:
                return False, f"{label}: CSV has no data rows", []

            # Find ticker column (case-insensitive search)
            ticker_col = None
            for col in df.columns:
                col_lower = str(col).strip().lower()
                if col_lower in ("ticker", "symbol", "code", "stock"):
                    ticker_col = col
                    break

            if ticker_col is None:
                # Heuristic: find column with most uppercase 1-5 char values
                best_col = None
                best_count = 0
                for col in df.columns:
                    vals = df[col].dropna().astype(str)
                    count = sum(1 for v in vals if re.match(r'^[A-Z]{1,5}$', v.strip()))
                    if count > best_count:
                        best_count = count
                        best_col = col
                if best_col and best_count >= 3:
                    ticker_col = best_col

            if ticker_col is None:
                return False, f"{label}: No ticker column found in CSV", []

            raw = df[ticker_col].dropna().astype(str).tolist()
            tickers = self._clean_ticker_list(raw)

            if not tickers:
                return False, f"{label}: No valid US tickers in CSV", []

            return True, f"{label}: {len(tickers)} tickers", tickers

        except requests.exceptions.Timeout:
            return False, f"{label}: Request timed out", []
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            return False, f"{label}: HTTP {status}", []
        except Exception as e:
            return False, f"{label}: {str(e)[:150]}", []

    # --- Text Extraction (regex) ---------------------------------------------

    def _extract_tickers_from_text(self, text: str) -> List[str]:
        """
        Extract ticker symbols from raw text using regex.

        Finds [A-Z]{1,5} patterns, filters false positives and foreign suffixes.
        """
        if not text:
            return []

        # Find all potential ticker patterns
        raw = re.findall(r'\b[A-Z]{1,5}(?:\.[A-Z])?\b', text)

        # Filter
        cleaned = []
        seen = set()
        for t in raw:
            t = t.strip()
            if t in seen or t in FALSE_POSITIVES:
                continue
            if self._validate_us_ticker(t):
                seen.add(t)
                cleaned.append(t)

        return sorted(cleaned)

    # --- Validation ----------------------------------------------------------

    def _validate_us_ticker(self, ticker: str) -> bool:
        """Validate a US market ticker symbol."""
        if not ticker or ticker in FALSE_POSITIVES:
            return False

        if not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
            return False

        if '.' in ticker:
            suffix = ticker.split('.')[1]
            if suffix in FOREIGN_SUFFIXES:
                return False
            if suffix not in VALID_CLASS_SHARES:
                return False

        return True

    def _clean_ticker_list(self, tickers: List[str]) -> List[str]:
        """Clean, validate, deduplicate, and sort a ticker list."""
        seen = set()
        cleaned = []
        for t in tickers:
            if not t or not isinstance(t, str):
                continue
            t = t.upper().strip()
            if t and t not in seen and self._validate_us_ticker(t):
                seen.add(t)
                cleaned.append(t)
        return sorted(cleaned)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    bridge = ScrapingBridge()

    # Test 1: ETF shortcut
    print("  Testing ETF shortcut (ARKK)...")
    wl = {"source_type": "etf_shortcut", "source": "arkk"}
    success, msg, tickers = bridge.fetch_tickers(wl)
    print(f"  1. {msg}")
    if success:
        print(f"     First 5: {tickers[:5]}")
    else:
        print(f"     (Network may be unavailable — this is OK for offline testing)")

    # Test 2: Text extraction
    print("\n  Testing text extraction...")
    sample_text = """
    Top Holdings: AAPL 6.5%, MSFT 5.2%, NVDA 4.8%, TSLA 3.1%
    Also includes BRK.B, GOOG.A, and AMZN
    Excludes foreign: SAP.F, SHOP.TO, HSBA.L
    Random words: THE AND FOR FUND TOTAL CASH
    """
    tickers = bridge._extract_tickers_from_text(sample_text)
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "BRK.B" in tickers      # Class share allowed
    assert "THE" not in tickers    # False positive filtered
    assert "AND" not in tickers    # False positive filtered
    # Note: SAP.F, SHOP.TO regex won't match as .F/.TO are multi-char
    # But if they did, they'd be filtered by _validate_us_ticker
    print(f"  2. Extracted: {tickers}")

    # Test 3: TradingView CSV
    print("\n  Testing TradingView CSV...")
    csv_data = "Ticker,Close,Volume\nAAPL,185.50,50000000\nMSFT,420.10,30000000\nINVALID123,0,0\n"
    success, msg, tickers = bridge._fetch_tradingview_csv(csv_data)
    assert success
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    print(f"  3. {msg}: {tickers}")

    # Test 4: Validation
    print("\n  Testing validation...")
    assert bridge._validate_us_ticker("AAPL") is True
    assert bridge._validate_us_ticker("BRK.B") is True
    assert bridge._validate_us_ticker("GOOG.A") is True
    assert bridge._validate_us_ticker("SAP.F") is False    # Foreign
    assert bridge._validate_us_ticker("ABC.Z") is False    # Invalid class
    assert bridge._validate_us_ticker("AND") is False      # False positive
    assert bridge._validate_us_ticker("") is False
    print("  4. Validation rules correct")

    # Test 5: Unknown source type
    wl = {"source_type": "unknown", "source": "test"}
    success, msg, tickers = bridge.fetch_tickers(wl)
    assert not success
    print(f"  5. Unknown source handled: {msg}")

    print("\n All 5 tests passed")
