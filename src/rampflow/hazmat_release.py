"""Hazmat release: policy judgement and stowage under one point in time.

A release for a flight payload must answer two questions with *the same*
inputs:

1. Does the operational policy (e.g. a night-time loading restriction) allow
   the work at the planned instant, under the rule edition in force then?
2. Can the *entire* payload be stowed under the same rule edition's
   dangerous-goods segregation table?

Both questions are evaluated against one instant and one immutable
``RegulationEdition``.  When the payload can be re-stowed, the complete new
plan is returned; otherwise the whole release is rejected and nothing is
partially committed -- no item is silently left loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from .timeutil import utc
from .models import Decision
from .cargo import CargoManager, CargoPosition, CargoPlan, ULD
from .operational_policy import PolicyContext, PolicyEngine, PolicyRule
from .dangerous_goods import DEFAULT_INCOMPATIBLE_PAIRS


@dataclass(frozen=True)
class RegulationEdition:
    """One immutable version of the rules in force.

    ``policy_rules`` govern when work is allowed (e.g. night curfews) and
    ``incompatible_pairs`` is the segregation table used by stowage.  They
    travel together so policy and stowage can never silently diverge.
    """
    edition_id: str
    valid_from: datetime
    valid_until: datetime | None = None
    policy_rules: tuple[PolicyRule, ...] = ()
    incompatible_pairs: frozenset[frozenset[str]] = DEFAULT_INCOMPATIBLE_PAIRS

    def __post_init__(self):
        object.__setattr__(self, "valid_from", utc(self.valid_from))
        if self.valid_until is not None:
            object.__setattr__(self, "valid_until", utc(self.valid_until))
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("invalid regulation edition validity")
        if not self.edition_id:
            raise ValueError("edition id required")

    def in_force(self, at: datetime) -> bool:
        t = utc(at)
        if t < self.valid_from:
            return False
        return self.valid_until is None or t < self.valid_until


class RegulationBook:
    """Versioned set of regulation editions; resolves the edition at an instant."""

    def __init__(self, editions: Iterable[RegulationEdition] = ()):
        self.editions: dict[str, RegulationEdition] = {}
        for edition in editions:
            self.add(edition)

    def add(self, edition: RegulationEdition) -> None:
        if edition.edition_id in self.editions:
            raise ValueError(f"duplicate edition {edition.edition_id}")
        self.editions[edition.edition_id] = edition

    def edition_at(self, at: datetime) -> RegulationEdition:
        t = utc(at)
        candidates = [e for e in self.editions.values() if e.in_force(t)]
        if not candidates:
            raise LookupError("no regulation edition in force at that instant")
        return max(candidates, key=lambda e: e.valid_from)


@dataclass(frozen=True)
class ItemPolicyVerdict:
    uld_id: str
    allowed: bool
    matched_rules: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseResult:
    """All-or-nothing outcome of a hazmat release request."""
    flight_id: str
    accepted: bool
    code: str
    reason: str
    at: datetime
    edition_id: str | None
    plan: CargoPlan | None = None
    verdicts: tuple[ItemPolicyVerdict, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def as_decision(self) -> Decision:
        return Decision(self.accepted, self.code, self.reason, dict(self.details))


class HazmatReleasePlanner:
    """Evaluates policy and re-stows the full payload atomically.

    The planner is bound to a :class:`RegulationBook` and one fixed instant per
    release, so both the :class:`PolicyEngine` and the :class:`CargoManager`
    it derives are constructed from the same edition -- same point in time,
    same version of the rules.
    """

    def __init__(self, book: RegulationBook, positions: Iterable[CargoPosition]):
        self.book = book
        self.positions = tuple(positions)
        self.committed: dict[str, ReleaseResult] = {}

    def _evaluate_policy(self, edition: RegulationEdition, flight_id: str, airport: str,
                         terminal: str, at: datetime, timezone: str,
                         ulds: Sequence[ULD]) -> tuple[ItemPolicyVerdict, ...]:
        engine = PolicyEngine(edition.policy_rules)
        verdicts = []
        for uld in ulds:
            ctx = PolicyContext(
                operation="hazmat-load",
                flight_id=flight_id,
                airport=airport,
                terminal=terminal,
                at=at,
                attributes={"uld": uld.uld_id, "dangerous_groups": tuple(sorted(uld.dangerous_groups))},
                timezone=timezone,
            )
            decision = engine.evaluate(ctx)
            verdicts.append(ItemPolicyVerdict(
                uld_id=uld.uld_id,
                allowed=decision.allowed,
                matched_rules=decision.matched_rules,
                reasons=decision.reasons,
            ))
        return tuple(verdicts)

    def release(self, flight_id: str, ulds: Iterable[ULD], at: datetime, *,
                airport: str = "", terminal: str = "", timezone: str = "UTC",
                loader_rate_min: int = 4, replace_existing: bool = True) -> ReleaseResult:
        """Return a complete re-stowed plan or reject the entire payload.

        A rejection never mutates committed state and never returns a partial
        assignment; the caller must hold back every item in the payload.
        """
        at = utc(at)
        items = tuple(ulds)
        if not items:
            raise ValueError("empty payload")

        # One instant resolves exactly one edition; both sides share it.
        edition = self.book.edition_at(at)

        verdicts = self._evaluate_policy(edition, flight_id, airport, terminal, at, timezone, items)
        denied = [v for v in verdicts if not v.allowed]
        if denied:
            return ReleaseResult(
                flight_id=flight_id,
                accepted=False,
                code="policy-denied",
                reason="policy denies hazmat loading at this time",
                at=at,
                edition_id=edition.edition_id,
                plan=None,
                verdicts=verdicts,
                details={"denied": tuple(v.uld_id for v in denied),
                         "reasons": tuple(r for v in denied for r in v.reasons)},
            )

        manager = CargoManager(self.positions, edition.incompatible_pairs)
        try:
            # plan() either places every ULD or raises -- no partial plan escapes.
            plan = manager.plan(flight_id, items, at, loader_rate_min=loader_rate_min)
        except RuntimeError as exc:
            return ReleaseResult(
                flight_id=flight_id,
                accepted=False,
                code="stow-impossible",
                reason=f"full re-stow impossible, entire payload rejected: {exc}",
                at=at,
                edition_id=edition.edition_id,
                plan=None,
                verdicts=verdicts,
                details={"error": str(exc)},
            )

        by_id = {u.uld_id: u for u in items}
        verification = manager.verify(plan, by_id)
        if not verification.accepted:
            return ReleaseResult(
                flight_id=flight_id,
                accepted=False,
                code="stow-verification",
                reason=f"re-stowed plan failed verification: {verification.reason}",
                at=at,
                edition_id=edition.edition_id,
                plan=None,
                verdicts=verdicts,
                details=dict(verification.details),
            )

        commit = manager.commit(plan, by_id)
        if not commit.accepted:
            return ReleaseResult(
                flight_id=flight_id,
                accepted=False,
                code="stow-commit",
                reason=commit.reason,
                at=at,
                edition_id=edition.edition_id,
                plan=None,
                verdicts=verdicts,
                details=dict(commit.details),
            )

        # Only a fully planned, verified and accepted release becomes state.
        result = ReleaseResult(
            flight_id=flight_id,
            accepted=True,
            code="ok",
            reason="full hazmat payload re-stowed and released",
            at=at,
            edition_id=edition.edition_id,
            plan=plan,
            verdicts=verdicts,
            details={"assignments": dict(plan.assignments), "sequence": plan.sequence},
        )
        if not replace_existing and flight_id in self.committed:
            raise RuntimeError(f"existing release for {flight_id}")
        self.committed[flight_id] = result
        return result
