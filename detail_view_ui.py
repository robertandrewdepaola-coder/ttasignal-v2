"""UI shell for scanner detail view.

Keeps tab-selection and scroll behavior in one place so app.py can delegate
navigation-heavy rendering with fewer inline state mutations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import streamlit as st

from navigation_state import (
    KEY_DEFAULT_DETAIL_TAB,
    KEY_DETAIL_NAV_INTENT,
    consume_detail_tab_selector_pending,
    consume_detail_tab_with_lock,
    consume_detail_nav_intent,
    detail_selector_key_for_ticker,
    detail_tab_for_target,
)


def render_detail_view_shell(
    *,
    analysis: Any,
    render_signal_tab: Callable[[str, Any, Dict[str, Any], Any], None],
    render_chart_tab: Callable[[str, Any], None],
    render_ai_tab: Callable[[str, Any, Dict[str, Any], Any], None],
    render_chat_tab: Callable[[str, Any, Dict[str, Any], Any], None],
    render_trade_tab: Callable[[str, Any, Any], None],
) -> None:
    """Render detail header, tab selector, and route to tab content callbacks."""
    if not analysis:
        return

    ticker = analysis.ticker
    signal = analysis.signal
    rec = analysis.recommendation or {}

    # Auto-scroll anchor — when a ticker is clicked, scroll here.
    st.markdown('<div id="detail-anchor"></div>', unsafe_allow_html=True)

    # Auto-scroll JavaScript — fires once when a new ticker is selected.
    if st.session_state.pop("scroll_to_detail", False):
        import streamlit.components.v1 as components

        components.html(
            """<script>
            const el = window.parent.document.getElementById('detail-anchor');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            </script>""",
            height=0,
        )

    # Header with scroll-to-top button.
    hdr_col1, hdr_col2 = st.columns([8, 1])
    with hdr_col1:
        st.header(f"{ticker} — {rec.get('recommendation', 'SKIP')}")
    with hdr_col2:
        if st.button("⬆️ Top", key="scroll_top", help="Scroll to top"):
            st.session_state["_do_scroll_top"] = True
            st.rerun()

    # Scroll-to-top JS — fires on next render after button click.
    if st.session_state.pop("_do_scroll_top", False):
        import streamlit.components.v1 as components

        components.html(
            """<script>
            setTimeout(function() {
                var doc = window.parent.document;
                // Try iframe parent scroll
                doc.querySelectorAll('[data-testid="stAppViewContainer"], section.main, .main, [data-testid="stMain"]').forEach(function(el) {
                    el.scrollTop = 0;
                });
                window.parent.scrollTo(0, 0);
                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;
            }, 100);
            </script>""",
            height=0,
        )

    st.caption(rec.get("summary", ""))

    # Detail view selector (state-driven; deterministic across reruns).
    tab_defs = [
        ("📊 Signal", "signal"),
        ("📈 Chart", "chart"),
        ("🤖 AI Intel", "ai"),
        ("💬 Ask AI", "chat"),
        ("💼 Trade", "trade"),
    ]
    tab_labels_by_key = {key: name for name, key in tab_defs}
    tab_keys = [key for _name, key in tab_defs]
    has_explicit_tab_intent = KEY_DEFAULT_DETAIL_TAB in st.session_state
    has_nav_intent = bool(st.session_state.get(KEY_DETAIL_NAV_INTENT))
    default_tab = consume_detail_tab_with_lock(
        st.session_state,
        ticker=ticker,
        fallback_tab=0,
        max_age_sec=60.0,
    )
    nav_target = consume_detail_nav_intent(
        st.session_state,
        ticker=ticker,
        max_age_sec=90.0,
    )
    if nav_target:
        default_tab = detail_tab_for_target(nav_target, fallback="chart")
    if default_tab < 0 or default_tab >= len(tab_defs):
        default_tab = 0
    target_tab_key = tab_defs[default_tab][1]
    _pending_target = consume_detail_tab_selector_pending(st.session_state, ticker=ticker)
    if _pending_target in tab_keys:
        target_tab_key = _pending_target
    selector_key = detail_selector_key_for_ticker(ticker)
    selected_tab_key = st.session_state.get(selector_key)
    if selected_tab_key not in tab_keys:
        st.session_state[selector_key] = target_tab_key
    elif has_explicit_tab_intent or has_nav_intent or bool(nav_target) or bool(_pending_target):
        st.session_state[selector_key] = target_tab_key

    selected_tab_key = st.radio(
        "Detail View",
        options=tab_keys,
        key=selector_key,
        format_func=lambda k: tab_labels_by_key.get(str(k), str(k)),
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_tab_key == "signal":
        render_signal_tab(ticker, signal, rec, analysis)
    elif selected_tab_key == "chart":
        render_chart_tab(ticker, signal)
    elif selected_tab_key == "ai":
        render_ai_tab(ticker, signal, rec, analysis)
    elif selected_tab_key == "chat":
        render_chat_tab(ticker, signal, rec, analysis)
    elif selected_tab_key == "trade":
        render_trade_tab(ticker, signal, analysis)
