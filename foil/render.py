"""Renderer: IR + fork_key -> exact Anthropic Messages payload.

Implements PREREGISTRATION.md §4 (fork semantics) and §6.2 (context layout):

    system  ->  world  ->  history  ->  [ ablatable report block ]  ->  query
    |______________ invariant, cache breakpoint here _____________|

The payload is content-addressed. Two runs with the same fork_key produce a
byte-identical request, which is the whole meaning of "deterministic replay"
(§4.2) -- responses are sampled and are never reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .ir import Episode, Span, SpanKind

#: Structured outputs give the low-cardinality action of §6.3 with a schema
#: the API enforces, so parsing is guaranteed rather than hopeful. Free-text
#: actions would need a classifier whose error rate becomes an unseparable
#: noise floor.
#:
#: Assistant prefill -- the cheaper mechanism -- is NOT available: it returns
#: 400 on every 4.6-and-later Opus/Sonnet-tier model, including the Phase 1
#: targets. Measured 2026-08-13, not assumed.
#:
#: Sampling is left at the model default. `temperature` is REJECTED at
#: non-default values on Sonnet 5 / Opus 5, so FOIL cannot sweep it; the
#: default is 1.0, which is the sampling the measurement needs, but it is a
#: fixed property of the model rather than a knob (§4.2 amendment).


class Operator(str, Enum):
    """Ablation operators (§4.4). These define DIFFERENT counterfactuals and
    their results are not comparable. Every reported residual states which one
    produced it."""

    DEL = "del"          # span removed entirely
    NULL = "null"        # explicit absence -- PRE-REGISTERED PRIMARY
    NEUTRAL = "neutral"  # well-formed but uninformative
    INVERT = "invert"    # asserted content negated; alignment probes only


def _ablate(span: Span, op: Operator) -> Span | None:
    name = span.source_id
    if op is Operator.DEL:
        return None
    if op is Operator.NULL:
        return Span(SpanKind.REPORT, f"Scout {name}: no report this round.", name, span.claim_id)
    if op is Operator.NEUTRAL:
        return Span(
            SpanKind.REPORT,
            f"Scout {name}: patrol completed; conditions unchanged from the last briefing.",
            name,
            span.claim_id,
        )
    if op is Operator.INVERT:
        t = span.text
        t = t.replace(" clear", " \x00").replace(" blocked", " clear").replace(" \x00", " blocked")
        return Span(SpanKind.REPORT, t, name, span.claim_id)
    raise ValueError(op)


@dataclass(frozen=True)
class ForkKey:
    """Fully determines a request (§4.1)."""

    episode_id: str
    decision_index: int
    coalition: frozenset[str]
    operator: Operator
    render_order: tuple[str, ...]
    model: str

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "coalition": sorted(self.coalition),
            "operator": self.operator.value,
            "render_order": list(self.render_order),
            "model": self.model,
        }


def action_schema(actions: tuple[str, ...]) -> dict:
    """JSON schema for the action. `enum` pins the support to the action set.

    No numeric bounds on `confidence`: the structured-output validator does not
    support `minimum`/`maximum`, so the range is stated in the query prose and
    treated as advisory. Confidence is not used by the Phase 1 metric.
    """
    return {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": list(actions)},
            "confidence": {"type": "number"},
        },
        "required": ["route", "confidence"],
        "additionalProperties": False,
    }


def render(ep: Episode, key: ForkKey, max_tokens: int = 256) -> dict:
    """Build the exact request body for one fork."""
    if set(key.render_order) != set(ep.source_ids):
        raise ValueError("render_order must be a permutation of the episode's sources")

    by_id = {s.source_id: s for s in ep.reports}

    # Invariant prefix: system + world (+ history when present).
    prefix = ep.invariant_prefix()
    system_blocks = [{"type": "text", "text": "\n\n".join(s.text for s in prefix)}]
    # Cache breakpoint at the end of the longest invariant run. Whether it
    # actually engages depends on the model's minimum cacheable prefix; the
    # executor reports measured cache tokens rather than assuming a saving.
    system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

    lines: list[str] = []
    for sid in key.render_order:
        span = by_id[sid]
        if sid not in key.coalition:
            ablated = _ablate(span, key.operator)
            if ablated is None:
                continue
            span = ablated
        lines.append(span.text)

    query = next(s for s in ep.spans if s.kind is SpanKind.QUERY)
    user_text = "\n".join(lines) + "\n\n" + query.text

    return {
        "model": key.model,
        "max_tokens": max_tokens,
        # Thinking is ON by default on Sonnet 5 / Opus 5, and max_tokens caps
        # thinking AND response text together. Phase 0 disables it: the nulls
        # measure harness noise, not epistemics, and a thinking listener is a
        # different (and far more expensive) object of study. Phase 1 must
        # decide this explicitly -- see the §7 amendment.
        "thinking": {"type": "disabled"},
        "output_config": {"format": {"type": "json_schema", "schema": action_schema(ep.actions)}},
        "system": system_blocks,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
    }


def request_hash(body: dict) -> str:
    """Content address for the rendered request (§4.2)."""
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def parse_action(text: str, actions: tuple[str, ...]) -> tuple[str | None, float | None]:
    """Recover (action, confidence) from a structured-output response.

    The schema guarantees valid JSON, so a parse failure means something else
    went wrong -- a truncated response (`stop_reason: max_tokens`) or a
    refusal. Unparseable samples are counted and reported, never silently
    dropped: a drifting parse rate is a noise source and has to stay visible.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    route = obj.get("route")
    conf = obj.get("confidence")
    if route not in actions:
        return None, None
    return route, (float(conf) if isinstance(conf, (int, float)) else None)
