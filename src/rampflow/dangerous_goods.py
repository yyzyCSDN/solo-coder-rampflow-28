from __future__ import annotations

from typing import Iterable

# Single source of truth for dangerous-goods segregation.  Every consumer
# (ULD cargo planning, compartment load control, load reconciliation and the
# hazmat release workflow) must consult this table (or an explicitly versioned
# edition of it) so that policy judgement and stowage never disagree about
# which dangerous classes may share a compartment.
DEFAULT_INCOMPATIBLE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"explosive", "flammable"}),
    frozenset({"oxidizer", "flammable"}),
    frozenset({"radioactive", "live-animal"}),
    frozenset({"acid", "cyanide"}),
})


def pair_conflict(a: str, b: str,
                  pairs: frozenset[frozenset[str]] = DEFAULT_INCOMPATIBLE_PAIRS) -> bool:
    """True when two dangerous-goods groups may not share a compartment."""
    if not a or not b:
        return False
    return frozenset({a, b}) in pairs


def groups_conflict(a: Iterable[str], b: Iterable[str],
                    pairs: frozenset[frozenset[str]] = DEFAULT_INCOMPATIBLE_PAIRS) -> bool:
    """True when any cross-product of the two group sets is incompatible."""
    return any(pair_conflict(x, y, pairs) for x in a for y in b)
