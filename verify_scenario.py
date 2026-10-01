"""Scenario verification for the B13 hazmat/night release fix."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

from rampflow.cargo import ULD, CargoPosition
from rampflow.operational_policy import PolicyRule, Effect
from rampflow.dangerous_goods import DEFAULT_INCOMPATIBLE_PAIRS
from rampflow.hazmat_release import RegulationBook, RegulationEdition, HazmatReleasePlanner
from rampflow.workflows.control import RampWorkflowControl
from rampflow.workflows.control import WorkflowRejected

SHA = ZoneInfo("Asia/Shanghai")
night = PolicyRule(
    "NIGHT-HAZMAT", 100, frozenset({"hazmat-load"}), Effect.DENY,
    "night curfew: no hazmat loading 23:00-06:00 local",
    start_local=time(23, 0), end_local=time(6, 0), airport="ZBAA",
)
day = PolicyRule(
    "DAY", 1, frozenset({"hazmat-load"}), Effect.ALLOW,
    "daytime hazmat loading permitted", airport="ZBAA",
)
edition = RegulationEdition(
    "2026.10", datetime(2026, 9, 1, tzinfo=SHA), None,
    policy_rules=(night, day),
    incompatible_pairs=DEFAULT_INCOMPATIBLE_PAIRS,
)
book = RegulationBook([edition])

# Two ULD decks: F (forward) and A (aft), each with two positions that are not
# declared adjacent across decks, so incompatible classes can be split.
positions = [
    CargoPosition("F1L", frozenset({"LD3", "M1"}), 5000, "F", "1", frozenset({"F1R"})),
    CargoPosition("F1R", frozenset({"LD3", "M1"}), 5000, "F", "2", frozenset({"F1L"})),
    CargoPosition("A1L", frozenset({"LD3", "M1"}), 5000, "A", "1", frozenset({"A1R"})),
    CargoPosition("A1R", frozenset({"LD3", "M1"}), 5000, "A", "2", frozenset({"A1L"})),
]

# Payload: two pairs of dangerous goods that must not share a compartment:
#   explosive + flammable  and  oxidizer + flammable
ulds = [
    ULD("U-EXPL", "LD3", 100, 800, "PEK", dangerous_groups=frozenset({"explosive"})),
    ULD("U-FLAM", "LD3", 100, 700, "PEK", dangerous_groups=frozenset({"flammable"})),
    ULD("U-OXID", "LD3", 100, 600, "PEK", dangerous_groups=frozenset({"oxidizer"})),
    ULD("U-BENIGN", "LD3", 100, 500, "PEK"),
]

print("== 1) 凌晨 00:30 本地时间作业 (旧逻辑会按 UTC 16:30 当成白天放行) ==")
planner = HazmatReleasePlanner(book, positions)
at_night = datetime(2026, 10, 1, 0, 30, tzinfo=SHA)
r_night = planner.release("CA1801", ulds, at_night, airport="ZBAA", terminal="T3", timezone="Asia/Shanghai")
print("accepted:", r_night.accepted, "| code:", r_night.code, "| edition:", r_night.edition_id)
assert not r_night.accepted and r_night.code == "policy-denied"
assert r_night.plan is None and "CA1801" not in planner.committed

print("\n== 2) 白天 10:30 本地时间：同一时点/同一版规则，给完整新分舱 ==")
at_day = datetime(2026, 10, 1, 10, 30, tzinfo=SHA)
r = planner.release("CA1801", ulds, at_day, airport="ZBAA", terminal="T3", timezone="Asia/Shanghai")
print("accepted:", r.accepted, "| code:", r.code)
for uid, pid in sorted(r.plan.assignments.items()):
    print(f"  {uid:10s} -> {pid}")
assigned = r.plan.assignments
assert r.accepted and len(assigned) == 4
# incompatible groups must not end up in the same position
pos_of = {u.uld_id: p for u, p in ((u, assigned[u.uld_id]) for u in ulds)}
assert assigned["U-EXPL"] != assigned["U-FLAM"]
assert assigned["U-OXID"] != assigned["U-FLAM"]
assert len(set(assigned.values())) == 4  # every position unique

print("\n== 3) 无法重分舱时整份拒绝，不静默留下任何货物 ==")
# Only one non-adjacent... force impossibility: two adjacent positions only.
tight = [
    CargoPosition("X1", frozenset({"LD3"}), 5000, "M", "1", frozenset({"X2"})),
    CargoPosition("X2", frozenset({"LD3"}), 5000, "M", "2", frozenset({"X1"})),
]
planner2 = HazmatReleasePlanner(book, tight)
r2 = planner2.release("CA1802", ulds, at_day, airport="ZBAA", terminal="T3", timezone="Asia/Shanghai")
print("accepted:", r2.accepted, "| code:", r2.code, "| reason:", r2.reason)
assert not r2.accepted and r2.code == "stow-impossible"
assert r2.plan is None and planner2.committed == {}

print("\n== 4) 被政策拒绝的载荷不会对外发布 committed 事件 (工作流层) ==")
control = RampWorkflowControl()
try:
    control.commit_hazmat_load(
        "OP-1", "CA1801", at_night,
        lambda: (_ for _ in ()).throw(RuntimeError("should never run")),
        requirements=(r_night.as_decision(),),
    )
    raise AssertionError("expected rejection")
except WorkflowRejected as e:
    print("workflow rejected:", e)
assert control.stream.head == 0 and control.receipts == {}

print("\nALL SCENARIO CHECKS PASSED")
