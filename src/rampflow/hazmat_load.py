from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from .cargo import CargoManager, CargoPlan, ULD
from .hazmat import HazmatRulebook
from .models import Decision
from .operational_policy import PolicyContext, PolicyDecision, PolicyEngine
from .timeutil import utc


@dataclass(frozen=True)
class HazmatLoadReview:
    """Whole-load verdict. Either a complete plan or an all-items rejection."""

    accepted: bool
    decision: Decision
    plan: CargoPlan | None
    evaluated_at: datetime
    policy: PolicyDecision


class HazmatLoadCoordinator:
    """Runs policy gating and stowage against one instant and one rule edition.

    The same timezone-aware `at` is fed to the policy engine and the cargo
    loader, and the policy engine and the cargo manager must advertise the same
    hazmat edition. A stowage failure rejects the whole request: no partial
    assignment is ever returned or committed.
    """

    def __init__(
        self,
        policy: PolicyEngine,
        cargo: CargoManager,
        rulebook: HazmatRulebook,
        operation: str = "cargo-load",
    ):
        if policy.edition and rulebook.version and policy.edition != rulebook.version:
            raise ValueError("policy and stowage configured with different hazmat editions")
        if cargo.rulebook.version != rulebook.version:
            raise ValueError("cargo manager configured with a different hazmat rulebook")
        self.policy = policy
        self.cargo = cargo
        self.rulebook = rulebook
        self.operation = operation

    def review(
        self,
        flight_id: str,
        airport: str,
        terminal: str,
        at: datetime,
        ulds: Iterable[ULD],
        attributes: Mapping[str, object] | None = None,
        loader_rate_min: int = 4,
    ) -> HazmatLoadReview:
        when = utc(at)
        items = tuple(ulds)
        ctx = PolicyContext(
            self.operation,
            flight_id,
            airport,
            terminal,
            when,
            dict(attributes or {}),
        )
        gated = self.policy.evaluate(ctx)
        if not gated.allowed:
            why = "; ".join(gated.reasons) if gated.reasons else "policy denied"
            decision = Decision(
                False,
                "policy-denied",
                f"hazmat load rejected by policy at {when.isoformat()}: {why}",
                {"reasons": gated.reasons, "matched_rules": gated.matched_rules,
                 "hazmat_version": self.rulebook.version},
            )
            return HazmatLoadReview(False, decision, None, when, gated)

        try:
            plan = self.cargo.plan(
                flight_id, items, when, loader_rate_min=loader_rate_min
            )
        except RuntimeError as exc:
            # Complete restow impossible -> reject the entire load rather than
            # silently leaving any of the items behind.
            decision = Decision(
                False,
                "no-stowage",
                f"whole hazmat load rejected: {exc}",
                {"hazmat_version": self.rulebook.version},
            )
            return HazmatLoadReview(False, decision, None, when, gated)

        verified = self.cargo.verify(plan, {u.uld_id: u for u in items})
        if not verified.accepted:
            decision = Decision(
                False,
                verified.code,
                f"whole hazmat load rejected: {verified.reason}",
                {**verified.details, "hazmat_version": self.rulebook.version},
            )
            return HazmatLoadReview(False, decision, None, when, gated)

        decision = Decision(
            True,
            "ok",
            "complete hazmat stowage plan produced",
            {"hazmat_version": plan.hazmat_version,
             "load_window": (plan.load_window.start.isoformat(), plan.load_window.end.isoformat())},
        )
        return HazmatLoadReview(True, decision, plan, when, gated)
