"""PID: permutation-instability detection (PREREGISTRATION-PID.md).

A different task family and a different model from the run the effect was
noticed on. Evidence-passage QA rather than scripted-scout routing; the
permutable units are prose passages, and answering requires a 2-hop read rather
than elimination.

Item determinacy is asserted analytically at generation time. No item reaches a
model unless exactly one option is entailed by the passages.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

# Fictional entities: parametric knowledge must not be able to answer these.
VESSELS = ["Kestrel-7", "Ondrail", "Marisole", "Thane-IV", "Verrow", "Astrolab",
           "Cindral", "Pellamar", "Quorum-3", "Halvane", "Bettrix", "Sorrel-9"]
PORTS = ["Kaldis", "Prienne", "Ostrova", "Merrow Bay", "Tamsin Reach", "Vellin"]
CARGOS = ["kelp resin", "cobalt ore", "glass fibre", "salt cedar", "iron black", "amber wax"]
INSPECTORS = ["Adaku", "Renshaw", "Oyelaran", "Fitch", "Nakamura", "Brindle"]


@dataclass(frozen=True)
class Item:
    item_id: str
    passages: tuple[str, ...]
    question: str
    options: tuple[str, ...]
    answer: str


def _entailed(chain: dict, options: tuple[str, ...]) -> list[str]:
    """Options entailed by the 2-hop chain. Must be exactly one.

    The chain is: vessel -> port (hop 1), port -> inspector (hop 2). The
    question asks which inspector cleared the vessel, so only the inspector at
    the vessel's port is entailed. Distractor passages name other
    port/inspector pairs, which entail nothing about this vessel.
    """
    return [o for o in options if o == chain["inspector"]]


def make_item(seed: int) -> Item | None:
    rng = random.Random(seed)
    vessel = rng.choice(VESSELS)
    ports = rng.sample(PORTS, 4)
    inspectors = rng.sample(INSPECTORS, 4)
    cargo = rng.choice(CARGOS)

    home, *others = ports
    chain = {"vessel": vessel, "port": home, "inspector": inspectors[0]}

    # Two decisive passages, separated so neither alone answers the question.
    decisive = [
        f"The {vessel} completed her refit at {home} and has berthed there since.",
        f"All vessels berthed at {home} are cleared by inspector {inspectors[0]}.",
    ]
    # Distractors: true, well-formed, and irrelevant to the asked chain.
    # Every non-answer option is named by some passage, so "appears in the
    # evidence at all" cannot be used as a shortcut heuristic.
    distractors = [
        f"Inspector {inspectors[1]} was reassigned from {others[0]} last season.",
        f"The {vessel} last carried {cargo}; inspector {inspectors[3]} logged the manifest.",
        f"Vessels berthed at {others[1]} are cleared by inspector {inspectors[2]}.",
    ]

    answer = chain["inspector"]
    # Options are shuffled INDEPENDENTLY of the answer. Without this the
    # correct answer is always options[0] -- and the JSON-schema enum would
    # publish that ordering to the model, making the item answerable at 100%
    # by always picking the first option.
    options = list(inspectors[:4])
    rng.shuffle(options)
    options = tuple(options)

    # Analytic determinacy check -- before any model sees this item.
    ent = _entailed(chain, options)
    if len(ent) != 1 or ent[0] != answer:
        return None
    # No distractor may name the vessel together with a different port.
    if any(vessel in d and any(p in d for p in others) for d in distractors):
        return None

    passages = tuple(decisive + distractors)
    question = f"Which inspector clears the {vessel}?"
    return Item(f"pid-{seed}", passages, question, options, answer)


SYSTEM = (
    "You answer questions from a set of evidence passages. "
    "Use only the passages provided. Exactly one option is correct."
)


def render_item(it: Item, order: tuple[int, ...], model: str) -> dict:
    """Build the request. Passage order is the manipulated variable."""
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
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }


def permutations_for(it: Item, m: int, seed: int) -> list[tuple[int, ...]]:
    """Canonical order plus m-1 distinct random permutations, seeded."""
    n = len(it.passages)
    canonical = tuple(range(n))
    rng = random.Random(seed)
    pool = [p for p in itertools.permutations(range(n)) if p != canonical]
    rng.shuffle(pool)
    return [canonical] + pool[: m - 1]
