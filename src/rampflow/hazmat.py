from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Single edition shared by policy gating, ULD stowage and compartment allocation.
# Bump this string whenever the segregation table changes so that a plan planned
# under one edition can never be silently verified under another.
HAZMAT_EDITION = "2026.1"

# Canonical segregation table. Pairs in this set must never share a compartment
# (nor occupy adjacent positions). This is the union every domain module must use;
# do not re-declare subsets inside cargo or load-control code.
DEFAULT_INCOMPATIBLE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"explosive", "flammable"}),
    frozenset({"oxidizer", "flammable"}),
    frozenset({"radioactive", "live-animal"}),
    frozenset({"acid", "cyanide"}),
})


@dataclass(frozen=True)
class HazmatRulebook:
    """Versioned dangerous-goods segregation rules.

    One instance is meant to be shared by policy judgement and stowage so that
    both sides answer with the same edition for the same point in time.
    """

    version: str
    incompatible_pairs: frozenset[frozenset[str]] = DEFAULT_INCOMPATIBLE_PAIRS

    def conflicting_pair(
        self, groups_a: Iterable[str], groups_b: Iterable[str]
    ) -> tuple[str, str] | None:
        """Return an incompatible group pair present across the two group sets.

        Passing the same group set twice also detects a single declaration that
        itself carries two incompatible groups.
        """
        set_b = frozenset(groups_b)
        for x in groups_a:
            for y in set_b:
                if x == y:
                    continue
                if frozenset({x, y}) in self.incompatible_pairs:
                    return (x, y)
        return None

    def conflict(self, groups_a: Iterable[str], groups_b: Iterable[str]) -> bool:
        return self.conflicting_pair(groups_a, groups_b) is not None


HAZMAT_RULEBOOK = HazmatRulebook(HAZMAT_EDITION, DEFAULT_INCOMPATIBLE_PAIRS)
