import unittest
from datetime import datetime, timezone, time
from dataclasses import replace

from rampflow.hazmat import HAZMAT_EDITION, HAZMAT_RULEBOOK
from rampflow.operational_policy import PolicyRule, PolicyContext, PolicyEngine, Effect
from rampflow.cargo import CargoManager, CargoPosition, ULD
from rampflow.hazmat_load import HazmatLoadCoordinator
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected
from rampflow.models import Decision

UTC = timezone.utc
AT = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def night_rule():
    return PolicyRule(
        "NIGHT", 100, frozenset({"cargo-load"}), Effect.DENY,
        "night curfew", start_local=time(22, 0), end_local=time(6, 0),
    )


def policy_engine():
    return PolicyEngine([night_rule()], airport_zones={"PVG": "Asia/Shanghai"}, edition=HAZMAT_EDITION)


def positions():
    return [
        CargoPosition("A1", frozenset({"M1"}), 1000, "lower", "A", adjacent=frozenset({"A2"})),
        CargoPosition("A2", frozenset({"M1"}), 1000, "lower", "A", adjacent=frozenset({"A1"})),
        CargoPosition("B1", frozenset({"M1"}), 1000, "lower", "B"),
    ]


def hazmat_load():
    return [
        ULD("U1", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"acid"}), priority=1),
        ULD("U2", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"cyanide"}), priority=2),
    ]


class NightPolicyTests(unittest.TestCase):
    def test_overnight_window_matches_in_airport_local_time(self):
        engine = policy_engine()
        # 20:00 UTC = 04:00 in Shanghai -> inside 22:00-06:00
        denied = engine.evaluate(PolicyContext("cargo-load", "F", "PVG", "T", AT(20), {}))
        self.assertFalse(denied.allowed)
        self.assertIn("NIGHT", denied.matched_rules)
        # 02:00 UTC = 10:00 in Shanghai -> daytime, allowed
        allowed = engine.evaluate(PolicyContext("cargo-load", "F", "PVG", "T", AT(2), {}))
        self.assertTrue(allowed)
        self.assertNotIn("NIGHT", allowed.matched_rules)

    def test_window_boundaries_are_start_inclusive_end_exclusive(self):
        engine = policy_engine()
        self.assertFalse(engine.evaluate(PolicyContext("cargo-load", "F", "PVG", "T", AT(14), {})).allowed)
        end = datetime(2025, 12, 31, 22, 0, tzinfo=UTC)  # 06:00 Shanghai
        self.assertTrue(engine.evaluate(PolicyContext("cargo-load", "F", "PVG", "T", end, {})).allowed)

    def test_rule_zone_overrides_context_airport_zone(self):
        rule = PolicyRule("N", 100, frozenset({"cargo-load"}), Effect.DENY, "x",
                          start_local=time(22), end_local=time(6), zone="UTC")
        engine = PolicyEngine([rule])
        ctx = PolicyContext("cargo-load", "F", "PVG", "T", AT(23), {}, zone="Asia/Shanghai")
        self.assertFalse(engine.evaluate(ctx).allowed)

    def test_decision_carries_edition(self):
        self.assertEqual(policy_engine().evaluate(
            PolicyContext("cargo-load", "F", "PVG", "T", AT(2), {})).edition, HAZMAT_EDITION)


class SharedRulebookTests(unittest.TestCase):
    def test_all_four_canonical_pairs_detected(self):
        for a, b in (("explosive", "flammable"), ("oxidizer", "flammable"),
                     ("radioactive", "live-animal"), ("acid", "cyanide")):
            self.assertIsNotNone(HAZMAT_RULEBOOK.conflicting_pair((a,), (b,)), (a, b))

    def test_self_declared_incompatible_groups_detected(self):
        self.assertIsNotNone(HAZMAT_RULEBOOK.conflicting_pair(("acid", "cyanide"), ("acid", "cyanide")))


class CargoRestowTests(unittest.TestCase):
    def test_incompatible_load_restowed_into_separate_compartments(self):
        mgr = CargoManager(positions())
        plan = mgr.plan("F", hazmat_load(), AT(2))
        self.assertEqual(set(plan.assignments), {"U1", "U2"})
        pa = mgr.positions[plan.assignments["U1"]]
        pb = mgr.positions[plan.assignments["U2"]]
        self.assertNotEqual((pa.deck, pa.zone), (pb.deck, pb.zone))
        self.assertTrue(mgr.verify(plan, {u.uld_id: u for u in hazmat_load()}).accepted)
        self.assertEqual(plan.hazmat_version, HAZMAT_EDITION)

    def test_adjacent_only_positions_force_far_compartment(self):
        # A1/A2 adjacent in same zone; B free. Acid must not end up next to cyanide.
        mgr = CargoManager(positions())
        plan = mgr.plan("F", hazmat_load(), AT(2))
        in_b = {"U1", "U2"} & {u for u, p in plan.assignments.items() if p == "B1"}
        self.assertEqual(len(in_b), 1)

    def test_whole_load_rejected_when_restow_impossible(self):
        mgr = CargoManager([CargoPosition("A1", frozenset({"M1"}), 1000, "lower", "A")])
        with self.assertRaises(RuntimeError):
            mgr.plan("F", hazmat_load(), AT(2))
        self.assertEqual(mgr.plans, {})

    def test_verify_rejects_same_compartment_non_adjacent_conflict(self):
        # Two positions in the same deck/zone but NOT declared adjacent.
        mgr = CargoManager([
            CargoPosition("A1", frozenset({"M1"}), 1000, "lower", "A"),
            CargoPosition("A3", frozenset({"M1"}), 1000, "lower", "A"),
        ])
        mgr = CargoManager([
            CargoPosition("A1", frozenset({"M1"}), 1000, "lower", "A"),
            CargoPosition("A3", frozenset({"M1"}), 1000, "lower", "A"),
        ])
        # Not restowable (only one compartment) -> planner rejects whole load
        with self.assertRaises(RuntimeError):
            mgr.plan("F", [
                ULD("U1", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"flammable"})),
                ULD("U3", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"oxidizer"})),
            ], AT(2))
        # A hand-built conflicting plan is caught by verify even without adjacency.
        from rampflow.timeutil import Window
        from datetime import timedelta
        from rampflow.cargo import CargoPlan
        bogus = CargoPlan("F", {"U1": "A1", "U3": "A3"}, 200,
                          Window(AT(2), AT(2) + timedelta(minutes=8)), ("U1", "U3"), HAZMAT_EDITION)
        d = mgr.verify(bogus, {
            "U1": ULD("U1", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"flammable"})),
            "U3": ULD("U3", "M1", 100, 100, "PVG", dangerous_groups=frozenset({"oxidizer"})),
        })
        self.assertFalse(d.accepted)
        self.assertEqual(d.code, "hazmat-separation")

    def test_stale_edition_plan_refused(self):
        mgr = CargoManager(positions())
        plan = mgr.plan("F", hazmat_load(), AT(2))
        stale = replace(plan, hazmat_version="1999.0")
        d = mgr.verify(stale, {u.uld_id: u for u in hazmat_load()})
        self.assertFalse(d.accepted)
        self.assertEqual(d.code, "hazmat-edition")


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.mgr = CargoManager(positions())
        self.coord = HazmatLoadCoordinator(policy_engine(), self.mgr, HAZMAT_RULEBOOK)

    def test_daytime_complete_plan(self):
        review = self.coord.review("F", "PVG", "T", AT(2), hazmat_load())
        self.assertTrue(review.accepted)
        self.assertIsNotNone(review.plan)
        self.assertEqual(review.evaluated_at, AT(2))
        self.assertEqual(review.plan.load_window.start, AT(2))

    def test_night_rejects_whole_load_without_plan_or_commit(self):
        review = self.coord.review("F", "PVG", "T", AT(20), hazmat_load())
        self.assertFalse(review.accepted)
        self.assertIsNone(review.plan)
        self.assertEqual(review.decision.code, "policy-denied")
        self.assertIn("night curfew", review.decision.reason)
        self.assertEqual(self.mgr.plans, {})

    def test_unrestowable_load_rejected_entirely(self):
        mgr = CargoManager([CargoPosition("A1", frozenset({"M1"}), 1000, "lower", "A")])
        coord = HazmatLoadCoordinator(policy_engine(), mgr, HAZMAT_RULEBOOK)
        review = coord.review("F", "PVG", "T", AT(2), hazmat_load())
        self.assertFalse(review.accepted)
        self.assertEqual(review.decision.code, "no-stowage")
        self.assertIsNone(review.plan)

    def test_policy_and_stowage_share_one_instant(self):
        # 13:00 UTC = 21:00 local allowed; 14:00 UTC = 22:00 local denied.
        self.assertTrue(self.coord.review("F", "PVG", "T", AT(13), hazmat_load()).accepted)
        denied = self.coord.review("F", "PVG", "T", AT(14), hazmat_load())
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.evaluated_at, AT(14))

    def test_edition_mismatch_refused_at_construction(self):
        other = PolicyEngine([night_rule()], airport_zones={"PVG": "Asia/Shanghai"}, edition="2024.0")
        with self.assertRaises(ValueError):
            HazmatLoadCoordinator(other, self.mgr, HAZMAT_RULEBOOK)


class HazmatWorkflowTests(unittest.TestCase):
    def test_rejected_requirement_publishes_nothing(self):
        wc = RampWorkflowControl()
        with self.assertRaises(WorkflowRejected):
            wc.commit_hazmat_load("op-1", "F", AT(20), lambda: "x",
                                  requirements=(Decision(False, "policy-denied", "night"),))
        self.assertEqual(wc.stream.head, 0)
        self.assertNotIn("op-1", wc.receipts)

    def test_accepted_commit_publishes_after_action_and_is_idempotent(self):
        wc = RampWorkflowControl()
        calls = []
        r1 = wc.commit_hazmat_load("op-2", "F", AT(2), lambda: calls.append(1) or "ok")
        r2 = wc.commit_hazmat_load("op-2", "F", AT(2), lambda: calls.append(2) or "ok")
        self.assertIs(r1, r2)
        self.assertEqual(calls, [1])
        self.assertEqual(wc.stream.head, 1)
        self.assertEqual(r1.event.topic, "hazmat.policy.load.committed")


if __name__ == "__main__":
    unittest.main()
