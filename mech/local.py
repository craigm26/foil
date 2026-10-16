"""Local renderer: FOIL's exact stimulus, located token by token.

The API renderer (`foil.render.render`) produces a Messages payload. A local
model needs the same words through its own chat template, plus a map of which
token positions each source's report occupies -- the thing every white-box
operator addresses.

TEXT IS NEVER REBUILT HERE. The system and user strings come from
`foil.render.render`, and this module only locates them. A stimulus that
differed from the API one by a stray newline would be a different experiment,
so the line map is reconstructed independently and then checked against the
renderer's own output; a mismatch raises instead of proceeding.

The structured-output schema the API enforces is emulated with an assistant
prefill, `{"route": "`, after which each action is required to be exactly one
token. The action distribution is then the next-token distribution over those
tokens, renormalised: exact, with no sampling and no parse step. A tokenizer
that cannot give one token per action makes the model ineligible (structural
gate) rather than pushing the readout onto multi-token scoring, whose result
is a different quantity from what constrained decoding samples.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, replace

from foil.ir import Episode, SpanKind
from foil.render import ForkKey, Operator, _ablate, render

PREFILL = '{"route": "'


class ReadoutError(ValueError):
    """The tokenizer cannot give a one-token-per-action readout."""


class RenderMismatch(AssertionError):
    """The local line map disagrees with foil.render. Never proceed past this."""


def is_boundary_text(text: str) -> bool:
    """Punctuation or whitespace only -- the token classes the summary motif
    names (commas, periods, line and turn boundaries)."""
    s = text.strip()
    if not s:
        return True
    return all(unicodedata.category(ch).startswith("P") for ch in s)


@dataclass(frozen=True)
class SpanMap:
    """Where one rendered report line sits, in characters and in tokens.

    A token that straddles the line's end -- Qwen3 merges the final period
    with the following newline into one token -- belongs to the line. No token
    may belong to two lines; the structural gate checks that.
    """

    source_id: str
    text: str
    chars: tuple[int, int]
    tokens: tuple[int, ...]
    summary: tuple[int, ...]
    content: tuple[int, ...]

    @property
    def first(self) -> int:
        return self.tokens[0]

    @property
    def last(self) -> int:
        return self.tokens[-1]


@dataclass(frozen=True)
class LocalPrompt:
    episode_id: str
    order: tuple[str, ...]
    coalition: frozenset
    operator: str
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    spans: dict = field(hash=False)
    decision_pos: int = 0
    actions: tuple[str, ...] = ()
    action_ids: tuple[int, ...] = ()
    correct_index: int = 0
    region: tuple[str, ...] = ()
    answer_start: int = 0

    def digest(self) -> str:
        """Content address of the exact token sequence the model receives."""
        h = hashlib.sha256(",".join(map(str, self.input_ids)).encode())
        return h.hexdigest()[:16]

    def with_ids(self, ids: tuple[int, ...]) -> "LocalPrompt":
        """Same layout, different token ids. Used only by the leak test, which
        overwrites a span's tokens to prove nothing downstream can see them."""
        if len(ids) != len(self.input_ids):
            raise ValueError("the leak test requires an identical length")
        return replace(self, input_ids=tuple(ids), text="<token-edited>")


def chat_text(tok, system: str, user: str) -> tuple[str, int]:
    """Apply the model's chat template with thinking disabled, then prefill.

    Returns (text, answer_char): where the assistant turn begins. Thinking is
    disabled because FOIL disabled it (render.py): a thinking listener is a
    different object of study, and every result FOIL has is non-thinking.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {"tokenize": False, "enable_thinking": False}
    gen = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    nogen = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
    if not gen.startswith(nogen):
        raise RenderMismatch("chat template's generation prompt is not a suffix of the conversation")
    return gen + PREFILL, len(nogen)


def _lines(ep: Episode, order: tuple[str, ...], coalition: frozenset, op: Operator):
    by_id = {s.source_id: s for s in ep.reports}
    out = []
    for sid in order:
        span = by_id[sid]
        if sid not in coalition:
            span = _ablate(span, op)
            if span is None:
                continue
        out.append((sid, span.text))
    return out


def render_local(
    tok,
    ep: Episode,
    order: tuple[str, ...] | None = None,
    coalition=None,
    operator: Operator = Operator.NULL,
) -> LocalPrompt:
    order = tuple(order or ep.source_ids)
    coalition = frozenset(ep.source_ids if coalition is None else coalition)
    key = ForkKey(ep.episode_id, 0, coalition, operator, order, "local")
    body = render(ep, key)
    system = body["system"][0]["text"]
    user = body["messages"][0]["content"][0]["text"]

    lines = _lines(ep, order, coalition, operator)
    query = next(s.text for s in ep.spans if s.kind is SpanKind.QUERY)
    if "\n".join(t for _, t in lines) + "\n\n" + query != user:
        raise RenderMismatch(f"{ep.episode_id}: line map does not reproduce foil.render's user text")

    text, answer_char = chat_text(tok, system, user)
    for name, piece in (("user", user), ("system", system)):
        if text.count(piece) != 1:
            raise RenderMismatch(f"{name} text appears {text.count(piece)} times in the templated prompt")
    u0 = text.index(user)
    s0 = text.index(system)

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = tuple(enc["input_ids"])
    offs = tuple(tuple(o) for o in enc["offset_mapping"])

    def overlapping(a: int, b: int) -> tuple[int, ...]:
        return tuple(i for i, (x, y) in enumerate(offs) if y > x and x < b and y > a)

    spans: dict[str, SpanMap] = {}
    cur = u0
    line_ranges = []
    for sid, t in lines:
        a, b = cur, cur + len(t)
        toks = overlapping(a, b)
        summary = tuple(i for i in toks if is_boundary_text(text[offs[i][0]:offs[i][1]]))
        content = tuple(i for i in toks if i not in summary)
        spans[sid] = SpanMap(sid, t, (a, b), toks, summary, content)
        line_ranges.append((a, b, sid))
        cur = b + 1  # the "\n" between lines
    q0 = u0 + len(user) - len(query)

    region = []
    for i, (x, y) in enumerate(offs):
        if x >= answer_char and y > x:
            region.append("answer")
            continue
        lab = None
        for a, b, sid in line_ranges:
            if y > x and x < b and y > a:
                lab = f"report:{sid}"
                break
        if lab is None:
            if s0 <= x < s0 + len(system):
                lab = "system"
            elif q0 <= x < q0 + len(query):
                lab = "query"
            elif u0 <= x < u0 + len(user):
                lab = "sep"
            else:
                lab = "template"
        region.append(lab)
    answer_start = next((i for i, r in enumerate(region) if r == "answer"), len(ids) - 1)

    action_ids = []
    for a in ep.actions:
        ids2 = tok(text + a, add_special_tokens=False)["input_ids"]
        if tuple(ids2[: len(ids)]) != ids or len(ids2) != len(ids) + 1:
            raise ReadoutError(f"action {a!r} is not a single token after the prefill for this tokenizer")
        action_ids.append(ids2[-1])
    if len(set(action_ids)) != len(action_ids):
        raise ReadoutError("two actions share a token")
    for a, b in ((a, b) for a in ep.actions for b in ep.actions if a != b):
        if b.startswith(a):
            raise ReadoutError(f"action {a!r} is a prefix of {b!r}; one token cannot decide between them")

    return LocalPrompt(
        episode_id=ep.episode_id,
        order=order,
        coalition=coalition,
        operator=operator.value,
        text=text,
        input_ids=ids,
        offsets=offs,
        spans=spans,
        decision_pos=len(ids) - 1,
        actions=tuple(ep.actions),
        action_ids=tuple(action_ids),
        correct_index=list(ep.actions).index(ep.correct_action),
        region=tuple(region),
        answer_start=answer_start,
    )


def knock(prompt: LocalPrompt, source_id: str, part: str = "all") -> tuple[tuple[int, ...], int]:
    """Knockout spec for one source: (key positions to hide, first query that
    may not see them). Queries inside the span are untouched, so the span's own
    summary tokens still form normally -- only reads from AFTER the span are cut.

    parts: "all" | "summary" (punctuation) | "content" (everything else) |
    "final" (the last token only) | "inner" (punctuation except the last).
    Qwen3 fuses a report's closing period with the following line break into
    one token, so "summary" and "final" also hide the line structure; "inner"
    is the punctuation that carries no line break, which is how a reader can
    bound that confound.
    """
    sm = prompt.spans[source_id]
    keys = {
        "all": sm.tokens,
        "summary": sm.summary,
        "content": sm.content,
        "final": (sm.last,),
        "inner": tuple(i for i in sm.summary if i != sm.last),
    }[part]
    return tuple(keys), sm.last + 1


def shifted_positions(prompt: LocalPrompt, source_id: str, delta: int) -> list[int]:
    """Rotary positions with everything after `source_id`'s span moved by
    `delta`: the arrangement Ω_null would produce, with the report still
    visible. A negative delta makes the first moved tokens share positions
    with the span's last ones -- off-distribution, as an invisible slot is,
    and declared as such in the registration (§3.2)."""
    q0 = prompt.spans[source_id].last + 1
    return [i if i < q0 else i + delta for i in range(len(prompt.input_ids))]


def position_class(prompt: LocalPrompt, pos: int, token_text: str) -> str:
    """Descriptive label for a carrier's position (§7.3 location table).

    decision     the last prompt token, whose logits are the readout
    answer       other assistant-turn tokens (template opener, prefill)
    turn         chat-template tokens outside the assistant turn -- for a
                 carrier, the end of the user's turn ('<|im_end|>', '\\n'),
                 the two tokens H3b's caveat is about
    punctuation  punctuation or whitespace inside the user's text
    content      everything else
    """
    if pos == prompt.decision_pos:
        return "decision"
    if prompt.region[pos] == "answer":
        return "answer"
    if prompt.region[pos] == "template":
        return "turn"
    if is_boundary_text(token_text):
        return "punctuation"
    return "content"


# ---------------------------------------------------------------------------
# Claims re-derived from rendered text (structural and construct gates)
# ---------------------------------------------------------------------------

_LINE = re.compile(r"^Scout (\w+): (.*)\.$")
_CLAUSE = re.compile(r"^(.*) (?:is|are) (clear|blocked)$")


def claims_from_text(lines: list[str]) -> dict[str, dict[str, bool]]:
    """Parse rendered report lines back into {scout: {route: is_clear}}.

    The gates re-derive properties from what the model is shown, never from
    the generator's fields: a checker reading `case.truth` agrees with the
    generator by construction (METHODS §2, rule 1).
    """
    out: dict[str, dict[str, bool]] = {}
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        name, body = m.group(1), m.group(2)
        per: dict[str, bool] = {}
        for clause in body.split("; "):
            c = _CLAUSE.match(clause)
            if not c:
                raise ValueError(f"unparseable clause {clause!r}")
            for route in c.group(1).split(", "):
                per[route.strip()] = c.group(2) == "clear"
        out[name] = per
    return out


def identified_route(claims: dict[str, dict[str, bool]], routes: tuple[str, ...]) -> str | None:
    """The single route consistent with every claim, when claims are mutually
    consistent and pin one route; otherwise None."""
    clear = {r for per in claims.values() for r, ok in per.items() if ok}
    blocked = {r for per in claims.values() for r, ok in per.items() if not ok}
    if clear & blocked:
        return None
    if clear:
        return next(iter(clear)) if len(clear) == 1 else None
    remaining = [r for r in routes if r not in blocked]
    return remaining[0] if len(remaining) == 1 else None


def honest_variant(tok, ep: Episode) -> LocalPrompt:
    """Full information, no manipulation: the liar's report inverted to the
    truth (Ω_invert on the scout whose every claim is a lie). The construct
    gate asks whether the model reaches the truth from this."""
    liars = [s for s, r in ep.reliability.items() if r <= 0.0]
    if len(liars) != 1:
        raise ValueError("the construct variant needs exactly one always-lying scout (env3)")
    coalition = frozenset(ep.source_ids) - {liars[0]}
    return render_local(tok, ep, tuple(ep.source_ids), coalition, Operator.INVERT)
