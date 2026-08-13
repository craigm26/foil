"""Transcript IR.

An episode is a list of typed spans. Ablation addresses span IDs; prompts are
never edited as strings. This is the invariant that makes fork semantics
(PREREGISTRATION.md §4) well defined rather than a string-munging accident.

Span order in the IR is the CANONICAL order. The renderer may present report
spans in a different order (render_order), which is exactly what null N1
measures the cost of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SpanKind(str, Enum):
    SYSTEM = "system"
    WORLD = "world"
    HISTORY = "history"
    REPORT = "report"
    QUERY = "query"


#: Spans that never change under any fork. Everything before the first
#: non-invariant span can carry a cache breakpoint.
INVARIANT_KINDS = frozenset({SpanKind.SYSTEM, SpanKind.WORLD, SpanKind.HISTORY})


@dataclass(frozen=True)
class Span:
    kind: SpanKind
    text: str
    #: Present only on REPORT spans. Identifies the information source.
    source_id: str | None = None
    #: Present only on REPORT spans. Identifies the proposition asserted, so
    #: Phase 2 can attribute over claims rather than sources without a
    #: re-render. Unused in Phase 1.
    claim_id: str | None = None


@dataclass(frozen=True)
class Episode:
    """One seeded environment instance at one decision point."""

    episode_id: str
    spans: tuple[Span, ...]
    #: The action set presented to the listener. |A| <= 5 per §6.3.
    actions: tuple[str, ...]
    #: The environment-defined correct action for the hidden world state.
    correct_action: str
    #: Ground-truth reliability per source_id, used only by the normative
    #: module. The listener never sees this.
    reliability: dict[str, float] = field(default_factory=dict)
    #: Which sources reported on which actions. Drives the redundancy matrix
    #: that §2.3(c) flags as a sensitivity confound.
    coverage: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def reports(self) -> tuple[Span, ...]:
        return tuple(s for s in self.spans if s.kind is SpanKind.REPORT)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(s.source_id for s in self.reports)  # type: ignore[misc]

    def invariant_prefix(self) -> tuple[Span, ...]:
        """Longest leading run of spans that no fork can alter.

        The cache breakpoint goes at the end of this run. If it is shorter
        than the model's minimum cacheable prefix, caching will not engage at
        all -- the executor reports that rather than assuming a saving.
        """
        out: list[Span] = []
        for s in self.spans:
            if s.kind not in INVARIANT_KINDS:
                break
            out.append(s)
        return tuple(out)
