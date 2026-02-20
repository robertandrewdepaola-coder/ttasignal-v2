import unittest

from trade_finder_helpers import (
    build_planned_trade,
    build_trade_finder_selection,
    compute_trade_score,
    derive_support_stop_levels,
)


class TradeFinderHelperTests(unittest.TestCase):
    def test_build_trade_finder_selection_defaults(self):
        row = {
            "ticker": "strl",
            "price": 421.2,
            "suggested_stop_loss": 400.14,
            "suggested_target": 463.32,
            "ai_buy_recommendation": "Strong Buy",
            "risk_reward": 2.0,
        }

        out = build_trade_finder_selection(row, generated_at_iso="2026-02-18", run_id="TF_1")

        self.assertEqual(out["ticker"], "STRL")
        self.assertEqual(out["entry"], 421.2)
        self.assertEqual(out["trade_finder_run_id"], "TF_1")
        self.assertIn("support_price", out)
        self.assertIn("stop_basis", out)

    def test_build_planned_trade_maps_candidate(self):
        row = {
            "ticker": "AAPL",
            "price": 200,
            "suggested_entry": 201,
            "suggested_stop_loss": 194,
            "suggested_target": 216,
            "risk_reward": 2.14,
            "ai_buy_recommendation": "Buy",
            "rank_score": 7.8,
            "reason": "Breakout",
            "ai_rationale": "Weekly and monthly aligned",
        }

        plan = build_planned_trade(row, run_id="TF_2")

        self.assertEqual(plan.ticker, "AAPL")
        self.assertEqual(plan.entry, 201)
        self.assertEqual(plan.stop, 194)
        self.assertEqual(plan.trade_finder_run_id, "TF_2")
        self.assertEqual(plan.notes, "Weekly and monthly aligned")

    def test_compute_trade_score_rewards_ready_high_rr(self):
        strong = {
            "rank_score": 8.0,
            "ai_buy_recommendation": "Strong Buy",
            "risk_reward": 2.4,
            "earn_days": 20,
            "decision_card": {"execution_readiness": "READY"},
        }
        weak = {
            "rank_score": 8.0,
            "ai_buy_recommendation": "Watch Only",
            "risk_reward": 1.0,
            "earn_days": 2,
            "decision_card": {"execution_readiness": "WAIT"},
        }
        self.assertGreater(compute_trade_score(strong), compute_trade_score(weak))

    def test_derive_support_stop_levels_uses_support_when_available(self):
        out = derive_support_stop_levels(entry=100.0, current_stop=95.0, support_price=96.0)
        self.assertEqual(out["support_price"], 96.0)
        self.assertGreater(out["support_stop_price"], 0.0)
        self.assertLess(out["recommended_stop"], 100.0)
        self.assertIn("support", out["stop_basis"])


if __name__ == "__main__":
    unittest.main()
