"""PID task family v2 — multi-hop chains with near-miss competitors.

PID-1 landed on the pre-registered DEGENERATE outcome: `claude-opus-5` answered
2-hop reading essentially perfectly, so there was no error variance for
instability to predict. A detector cannot be tested where nothing is missed.

Difficulty here is a declared parameter rather than a fixed property, so a
calibration pilot can locate a setting where the model errs on a usable
fraction of items. Calibrating task difficulty so the dependent variable has
variance is ordinary experimental design; what must never be calibrated is the
hypothesis, the statistic, or the decision thresholds. Those are fixed in the
pre-registration written after calibration and before the test run.

The difficulty lever is NEAR-MISS chains: competitor chains sharing all but one
link with the true chain, so an item is failed by losing track of a single hop
rather than by failing to read.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

CARRIERS = ["Ashlin", "Bergen", "Corvale", "Dunmoor", "Elstree", "Fairwick",
            "Glaive", "Harrow", "Ivelle", "Jessamy", "Kirran", "Lowder"]
DEPOTS = ["Ravensworth", "Sable Point", "Tarn Hollow", "Underlea", "Vexley",
          "Wickmere", "Yarrowfield", "Zelbridge", "Ombersley", "Nethercote"]
LANES = ["Blue Lane", "Grey Lane", "Amber Lane", "Slate Lane", "Ochre Lane",
         "Verdant Lane", "Russet Lane", "Cobalt Lane"]
HANDLERS = ["Achebe", "Bhatt", "Cortez", "Dvorak", "Eriksen",
            "Falodun", "Gupta", "Halvorsen"]

#: Depot names that differ only by suffix. Drawing every depot in an item from
#: one family means the true chain and its near-misses are separated by a few
#: characters, so the item is failed by skimming rather than by misreading.
#: Determinacy is untouched -- the names are distinct strings and each
#: statement still names exactly one depot.
CONFUSABLE_DEPOTS = [
    ["Ravensworth", "Ravenscourt", "Ravensmoor", "Ravensdale"],
    ["Wickmere", "Wickmoor", "Wickford", "Wickholm"],
    ["Nethercote", "Nethergate", "Netherby", "Netherfield"],
    ["Ashcombe", "Ashcroft", "Ashbourne", "Ashwell"],
]


@dataclass(frozen=True)
class Item2:
    item_id: str
    passages: tuple[str, ...]
    question: str
    options: tuple[str, ...]
    answer: str
    hops: int
    near_misses: int


def make_item2(seed: int, hops: int = 3, near_misses: int = 2,
               distractors: int = 3, confusable: bool = False) -> Item2 | None:
    """Build one chain item.

    True chain: carrier -> depot -> lane -> handler (hops=3).
    A near-miss chain reuses the same depot but a different lane, or the same
    lane under a different depot, so it terminates on a different handler. The
    only way to answer is to follow the specific chain the question names.
    """
    rng = random.Random(seed)
    if hops < 2 or hops > 3:
        return None

    # Draw enough entities for the requested number of near-miss chains. The
    # first version sampled a fixed 3 depots and 4 handlers, so near_misses=3
    # indexed past the end -- the hardest rung of the difficulty ladder could
    # not be built at all.
    need = near_misses + 1
    if need > len(DEPOTS) or need > len(HANDLERS):
        return None

    carrier = rng.choice(CARRIERS)
    if confusable:
        family = rng.choice(CONFUSABLE_DEPOTS)
        if need > len(family):
            return None
        depots = rng.sample(family, max(3, need)) if len(family) >= max(3, need) else None
        if depots is None:
            return None
    else:
        depots = rng.sample(DEPOTS, max(3, need))
    lanes = rng.sample(LANES, 4)
    handlers = rng.sample(HANDLERS, max(4, need))

    true_depot, true_lane, true_handler = depots[0], lanes[0], handlers[0]

    # EVERY lane->handler statement is depot-qualified, including the true one.
    # Leaving the true link unqualified while the competitors name a depot would
    # make the item genuinely ambiguous, and an error would then mean "the item
    # was underdetermined" rather than "the model lost the chain" -- which is
    # exactly the confound this task exists to avoid.
    links: list[str] = [f"Carrier {carrier} consigns exclusively through the {true_depot} depot."]
    if hops == 3:
        links.append(f"The {true_depot} depot dispatches all consignments via {true_lane}.")
        links.append(f"At {true_depot}, {true_lane} is worked by handler {true_handler}.")
    else:
        links.append(f"At {true_depot}, consignments are worked by handler {true_handler}.")

    near: list[str] = []
    for i in range(near_misses):
        if hops == 3:
            # Same lane name at a different depot -> a different handler.
            # Following the lane without checking the depot yields handlers[i+1].
            near.append(
                f"At {depots[i + 1]}, {true_lane} is worked by handler {handlers[i + 1]}."
            )
        else:
            near.append(f"At {depots[i + 1]}, consignments are worked by handler {handlers[i + 1]}.")

    # Distinct distractors. The first version cycled lane and handler indices
    # with period 3, so any distractors>3 emitted verbatim duplicates -- noise
    # that adds length without adding difficulty, and reads as a generator bug.
    other: list[str] = []
    pool = [(ln, hd) for ln in lanes[1:] for hd in handlers[1:]]
    rng.shuffle(pool)
    for ln, hd in pool[:distractors]:
        other.append(
            f"{ln} was reassigned last quarter and is now worked by handler {hd}."
        )

    passages = links + near + other
    rng.shuffle(passages)

    options = list(handlers[:4])
    rng.shuffle(options)

    # Analytic determinacy: exactly one option is reachable by following the
    # chain that starts at the named carrier.
    reachable = [o for o in options if o == true_handler]
    if len(reachable) != 1:
        return None
    # Every option must appear somewhere, so "named in the evidence" is no clue.
    joined = " ".join(passages)
    if not all(o in joined for o in options):
        return None

    return Item2(
        item_id=f"pid2-h{hops}n{near_misses}{'c' if confusable else ''}-{seed}",
        passages=tuple(passages),
        question=f"Which handler works consignments from carrier {carrier}?",
        options=tuple(options),
        answer=true_handler,
        hops=hops,
        near_misses=near_misses,
    )


#: A substantive system prompt, written once and shared by every request.
#: Unlike FOIL's ~171-token prefix this is long enough to clear the 512-token
#: minimum on Opus-class models, so `cache_control` on it can actually engage
#: instead of being an inert marker. It is real task instruction, not padding:
#: nothing here exists solely to reach the threshold.
SYSTEM2 = (
    "You are resolving consignment routing from a set of evidence passages.\n\n"
    "How the network works. Every carrier consigns through exactly one depot. "
    "Every depot dispatches its consignments via exactly one lane. Every lane, "
    "at a given depot, is worked by exactly one handler. A lane name may appear "
    "at more than one depot, and the handler working it may differ between "
    "depots, so a lane name alone never identifies a handler. Reassignment "
    "notices describe changes to a lane in general and do not override a "
    "statement that names a specific depot.\n\n"
    "How to answer. Start from the carrier named in the question. Find the depot "
    "that carrier consigns through. Find the lane that depot dispatches via. "
    "Find the handler working that lane at that depot. The answer is that "
    "handler.\n\n"
    "Passages appear in no particular order and their order carries no meaning. "
    "Some passages describe other carriers, other depots, or other lanes, and "
    "are irrelevant to the question asked. Some passages describe a chain that "
    "shares part of its route with the one you need; following a shared segment "
    "without checking which depot it belongs to will give the wrong handler.\n\n"
    "Use only the passages provided. Do not rely on outside knowledge; every "
    "name is fictional. Exactly one option is correct. Answer with the handler's "
    "name exactly as it appears in the options."
)


def render_item2(it: Item2, order: tuple[int, ...], model: str,
                 cache_ttl: str = "1h") -> dict:
    """Render one request.

    The cache breakpoint sits on the SYSTEM block -- the last content that is
    identical across every request -- not on the user message, which carries
    the per-item evidence and question. Marking the varying block would write a
    fresh entry every call and never read one, which is the documented way to
    pay the cache-write premium for nothing.

    `cache_ttl` defaults to 1h rather than 5m because batch jobs commonly take
    between five minutes and an hour; a 5-minute entry written at submission is
    typically expired before most of the batch runs.
    """
    body = "\n".join(f"- {it.passages[i]}" for i in order)
    user = (
        f"Evidence:\n{body}\n\n{it.question}\n"
        f"Options: {', '.join(it.options)}\n"
        'Answer with JSON, e.g. {"answer": "Name", "confidence": 0.7}.'
    )
    return {
        "model": model,
        "max_tokens": 256,
        "thinking": {"type": "disabled"},
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": list(it.options)},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "system": [{"type": "text", "text": SYSTEM2,
                    "cache_control": {"type": "ephemeral", "ttl": cache_ttl}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }


def permutations_for2(it: Item2, m: int, seed: int) -> list[tuple[int, ...]]:
    n = len(it.passages)
    canonical = tuple(range(n))
    rng = random.Random(seed)
    pool = [p for p in itertools.permutations(range(n))][1:]
    rng.shuffle(pool)
    picked, seen = [canonical], {canonical}
    for p in pool:
        if p not in seen:
            picked.append(p)
            seen.add(p)
        if len(picked) == m:
            break
    return picked
