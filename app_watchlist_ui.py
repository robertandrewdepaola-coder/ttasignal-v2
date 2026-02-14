"""
TTA v2 Watchlist UI — Streamlit Sidebar Interface
===================================================

Renders the multi-watchlist management UI in the Streamlit sidebar.
Handles create, switch, refresh, add/remove tickers, delete, rollback.

Usage:
    from app_watchlist_ui import integrate_watchlist_ui

    watchlist_ui = integrate_watchlist_ui(journal_manager=jm)
    tickers = watchlist_ui.get_active_tickers()

Version: 1.0.0 (2026-02-14)
"""

import streamlit as st
from datetime import datetime
from typing import List, Optional

from watchlist_manager import WatchlistManager, MASTER_ID
from scraping_bridge import ScrapingBridge, ETF_SHORTCUTS


# =============================================================================
# WATCHLIST UI
# =============================================================================

class WatchlistUI:
    """
    Streamlit sidebar UI for multi-watchlist management.

    Provides create/switch/edit/refresh/delete controls.
    """

    def __init__(self, manager: WatchlistManager, bridge: ScrapingBridge):
        """Initialize UI with manager and bridge references."""
        self.manager = manager
        self.bridge = bridge

        # Initialize session state keys
        if "refresh_in_progress" not in st.session_state:
            st.session_state.refresh_in_progress = set()
        if "show_create_dialog" not in st.session_state:
            st.session_state.show_create_dialog = False
        if "wl_just_created" not in st.session_state:
            st.session_state.wl_just_created = None

    # --- Main Entry Point ----------------------------------------------------

    def render_sidebar(self) -> None:
        """Render the complete watchlist management UI in the sidebar."""
        with st.sidebar:
            st.markdown("### 📋 Watchlists")

            self._render_watchlist_selector()

            # Show create dialog or details
            if st.session_state.show_create_dialog:
                self._render_create_dialog()
            else:
                self._render_watchlist_details()

    # --- Watchlist Selector --------------------------------------------------

    def _render_watchlist_selector(self) -> None:
        """Dropdown to select active watchlist + Create New button."""
        watchlists = self.manager.get_all_watchlists()
        active = self.manager.get_active_watchlist()

        # Build labels with icons
        labels = []
        id_map = {}
        for wl in watchlists:
            if wl.get("is_system"):
                icon = "🔒"
            elif wl.get("type") == "auto":
                icon = "🔄"
            else:
                icon = "✏️"
            label = f"{icon} {wl['name']} ({len(wl.get('tickers', []))})"
            labels.append(label)
            id_map[label] = wl["id"]

        # Find current index
        active_label = None
        for label, wl_id in id_map.items():
            if wl_id == active["id"]:
                active_label = label
                break
        current_idx = labels.index(active_label) if active_label in labels else 0

        # Selector
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox(
                "Active Watchlist",
                labels,
                index=current_idx,
                key="wl_selector",
                label_visibility="collapsed",
            )
        with col2:
            if st.button("➕", key="wl_create_btn", help="Create new watchlist"):
                st.session_state.show_create_dialog = True
                st.rerun()

        # Handle selection change
        selected_id = id_map.get(selected)
        if selected_id and selected_id != active["id"]:
            self.manager.set_active_watchlist(selected_id)
            st.rerun()

    # --- Create Dialog -------------------------------------------------------

    def _render_create_dialog(self) -> None:
        """Form to create a new watchlist."""
        st.markdown("---")
        st.markdown("**Create New Watchlist**")

        with st.form("create_watchlist_form", clear_on_submit=True):
            name = st.text_input("Name", placeholder="e.g. ARKK Holdings")

            wl_type = st.radio(
                "Type",
                ["Manual", "Auto-Populate"],
                horizontal=True,
                key="create_type_radio",
            )

            source_type = None
            source = None

            if wl_type == "Auto-Populate":
                source_option = st.radio(
                    "Source",
                    ["ETF Holdings", "TradingView CSV", "Custom URL"],
                    horizontal=True,
                    key="create_source_radio",
                )

                if source_option == "ETF Holdings":
                    source_type = "etf_shortcut"
                    shortcuts = list(ETF_SHORTCUTS.keys())
                    shortcuts.append("Custom (enter URL)")
                    choice = st.selectbox("Select ETF", shortcuts, key="create_etf_select")

                    if choice == "Custom (enter URL)":
                        source_type = "etf_url"
                        source = st.text_input("ETF Holdings URL", key="create_etf_url")
                    else:
                        source = ETF_SHORTCUTS.get(choice, "")

                elif source_option == "TradingView CSV":
                    source_type = "tradingview"
                    uploaded = st.file_uploader(
                        "Upload CSV", type=["csv"], key="create_tv_upload"
                    )
                    if uploaded:
                        source = uploaded.getvalue().decode("utf-8", errors="replace")

                elif source_option == "Custom URL":
                    source_type = "custom_url"
                    source = st.text_input("URL to scrape", key="create_custom_url")

            # Form buttons
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Create", type="primary")
            with col2:
                cancelled = st.form_submit_button("Cancel")

            if cancelled:
                st.session_state.show_create_dialog = False
                st.rerun()

            if submitted:
                if not name or not name.strip():
                    st.error("Please enter a watchlist name")
                else:
                    try:
                        actual_type = "manual" if wl_type == "Manual" else "auto"
                        wl_id = self.manager.create_watchlist(
                            name=name.strip(),
                            wl_type=actual_type,
                            source_type=source_type,
                            source=source,
                        )

                        if wl_id:
                            self.manager.set_active_watchlist(wl_id)
                            st.session_state.show_create_dialog = False
                            st.session_state.wl_just_created = wl_id

                            # For TradingView CSV, process immediately
                            if source_type == "tradingview" and source:
                                success, msg, tickers = self.bridge._fetch_tradingview_csv(source)
                                if success:
                                    self.manager.update_tickers(wl_id, tickers, backup_old=False)

                            st.rerun()

                    except ValueError as e:
                        st.error(str(e))

    # --- Watchlist Details ---------------------------------------------------

    def _render_watchlist_details(self) -> None:
        """Show details of the active watchlist."""
        wl = self.manager.get_active_watchlist()
        ticker_count = len(wl.get("tickers", []))

        # Just-created notification
        if st.session_state.get("wl_just_created") == wl["id"]:
            if wl.get("type") == "auto" and ticker_count == 0:
                st.info("👆 Click **Refresh** to populate tickers")
            st.session_state.wl_just_created = None

        # Metrics row
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tickers", ticker_count)
        with col2:
            updated = wl.get("last_updated")
            if updated:
                try:
                    dt = datetime.fromisoformat(updated)
                    st.caption(f"Updated {dt.strftime('%b %d %H:%M')}")
                except (ValueError, TypeError):
                    pass

        # Auto-populate controls
        if wl.get("type") == "auto":
            self._render_auto_controls(wl)

        # Manual ticker controls
        self._render_manual_ticker_controls(wl)

        # View tickers
        self._render_ticker_list(wl)

        # Delete (non-system only)
        if not wl.get("is_system"):
            self._render_delete_controls(wl)

    # --- Auto-Populate Controls ----------------------------------------------

    def _render_auto_controls(self, wl: dict) -> None:
        """Refresh button, cooldown timer, rollback, stats."""
        wl_id = wl["id"]
        st.markdown("---")

        # Source info
        source_type = wl.get("source_type", "")
        source = wl.get("source", "")
        if source_type == "etf_shortcut":
            st.caption(f"Source: ETF `{source.upper()}`")
        elif source:
            st.caption(f"Source: {source[:50]}...")

        # Refresh button
        can_refresh, remaining = self.manager.can_refresh_auto(wl_id)
        is_busy = wl_id in st.session_state.refresh_in_progress

        if is_busy:
            st.info("⏳ Refresh in progress...")
        elif not can_refresh:
            st.progress(
                1.0 - (remaining / 60.0),
                text=f"Cooldown: {remaining}s remaining",
            )
        else:
            if st.button("🔄 Refresh Tickers", key=f"refresh_{wl_id}"):
                self._handle_refresh(wl)

        # Stats
        stats = wl.get("scraping_stats", {})
        total = stats.get("total_refreshes", 0)
        success_count = stats.get("successful_refreshes", 0)
        failed = stats.get("failed_refreshes", 0)
        if total > 0:
            st.caption(f"Refreshes: {success_count}✓ / {failed}✗ (of {total})")

        last_error = stats.get("last_error")
        if last_error:
            st.caption(f"⚠️ Last error: {last_error[:80]}")

        # Rollback
        backup = wl.get("last_backup")
        if backup and backup.get("tickers"):
            backup_count = len(backup["tickers"])
            backup_time = backup.get("timestamp", "unknown")
            if st.button(
                f"↩️ Rollback ({backup_count} tickers)",
                key=f"rollback_{wl_id}",
                help=f"Restore from {backup_time}",
            ):
                ok, msg = self.manager.rollback_tickers(wl_id)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

    def _handle_refresh(self, wl: dict) -> None:
        """Execute auto-refresh: fetch tickers, update watchlist, record stats."""
        wl_id = wl["id"]

        # Lock
        st.session_state.refresh_in_progress.add(wl_id)
        self.manager.record_refresh_request(wl_id)

        try:
            with st.spinner("Fetching tickers..."):
                success, msg, tickers = self.bridge.fetch_tickers(wl)

            if success and tickers:
                ok, update_msg, cleaned = self.manager.update_tickers(
                    wl_id, tickers, backup_old=True
                )
                if ok:
                    self.manager.record_refresh_success(wl_id)
                    st.success(f"✓ {update_msg}")
                else:
                    self.manager.record_refresh_failure(wl_id, update_msg)
                    st.error(update_msg)
            else:
                self.manager.record_refresh_failure(wl_id, msg)
                st.error(f"✗ {msg}")

        except Exception as e:
            self.manager.record_refresh_failure(wl_id, str(e))
            st.error(f"Refresh failed: {str(e)[:200]}")
        finally:
            st.session_state.refresh_in_progress.discard(wl_id)
            st.rerun()

    # --- Manual Ticker Controls ----------------------------------------------

    def _render_manual_ticker_controls(self, wl: dict) -> None:
        """Add/remove individual tickers."""
        wl_id = wl["id"]
        st.markdown("---")

        # Add ticker
        with st.form(f"add_ticker_form_{wl_id}", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_ticker = st.text_input(
                    "Add ticker",
                    placeholder="e.g. AAPL",
                    key=f"add_input_{wl_id}",
                    label_visibility="collapsed",
                )
            with col2:
                add_submitted = st.form_submit_button("➕")

            if add_submitted and new_ticker:
                ok, msg = self.manager.add_manual_ticker(wl_id, new_ticker.strip())
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

        # Remove ticker
        tickers = wl.get("tickers", [])
        if tickers:
            col1, col2 = st.columns([3, 1])
            with col1:
                to_remove = st.selectbox(
                    "Remove",
                    tickers,
                    key=f"remove_select_{wl_id}",
                    label_visibility="collapsed",
                )
            with col2:
                if st.button("🗑️", key=f"remove_btn_{wl_id}"):
                    ok, msg = self.manager.remove_manual_ticker(wl_id, to_remove)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()

    # --- Ticker List Display -------------------------------------------------

    def _render_ticker_list(self, wl: dict) -> None:
        """Show tickers in an expander."""
        tickers = wl.get("tickers", [])
        label = f"📋 View Tickers ({len(tickers)})"

        with st.expander(label, expanded=False):
            if tickers:
                st.text_area(
                    "Tickers",
                    value=", ".join(tickers),
                    height=100,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"ticker_display_{wl['id']}",
                )
            else:
                st.caption("No tickers yet")

            # Show overrides for auto watchlists
            if wl.get("type") == "auto":
                additions = wl.get("manual_additions", [])
                exclusions = wl.get("manual_exclusions", [])
                if additions:
                    st.caption(f"➕ Manual adds: {', '.join(additions)}")
                if exclusions:
                    st.caption(f"➖ Manual removes: {', '.join(exclusions)}")

    # --- Delete Controls -----------------------------------------------------

    def _render_delete_controls(self, wl: dict) -> None:
        """Delete watchlist with confirmation checkbox."""
        st.markdown("---")
        confirm = st.checkbox(
            f"Delete '{wl['name']}'",
            key=f"delete_confirm_{wl['id']}",
        )
        if confirm:
            if st.button(
                "🗑️ Confirm Delete",
                key=f"delete_btn_{wl['id']}",
                type="primary",
            ):
                ok, msg = self.manager.delete_watchlist(wl["id"])
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

    # --- Public API ----------------------------------------------------------

    def get_active_tickers(self) -> List[str]:
        """Return tickers from the active watchlist."""
        return self.manager.get_active_watchlist().get("tickers", [])


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def integrate_watchlist_ui(journal_manager=None) -> WatchlistUI:
    """
    One-line integration for existing app.py.

    Creates/retrieves WatchlistManager and ScrapingBridge from session state.
    Renders sidebar UI. Returns WatchlistUI for ticker access.

    Usage:
        watchlist_ui = integrate_watchlist_ui(journal_manager=jm)
        tickers = watchlist_ui.get_active_tickers()
    """
    # Initialize managers in session state (survive reruns)
    if "watchlist_manager" not in st.session_state:
        st.session_state.watchlist_manager = WatchlistManager()

    if "scraping_bridge" not in st.session_state:
        st.session_state.scraping_bridge = ScrapingBridge()

    manager = st.session_state.watchlist_manager
    bridge = st.session_state.scraping_bridge

    # Create UI
    ui = WatchlistUI(manager=manager, bridge=bridge)
    ui.render_sidebar()

    # Optional: compatibility bridge for JournalManager migration
    if journal_manager is not None:
        if "watchlist_bridge" not in st.session_state:
            from watchlist_bridge import WatchlistBridge
            st.session_state.watchlist_bridge = WatchlistBridge(manager, journal_manager)

    return ui
