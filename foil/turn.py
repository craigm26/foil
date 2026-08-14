"""TURN: hidden-profile deliberation with speaking order as the manipulation.

Implements PREREGISTRATION-TURN.md.

The admissibility requirement is checked ANALYTICALLY at generation time by a
fixed reference scorer -- shared facts alone must favour a wrong candidate, and
shared plus private must uniquely favour the true one. Not verifying an
environment's defining property before spending is what cost FOIL's Phase 0 two
protocols; here it is a hard gate rather than an assumption.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

CANDIDATES = ["Renwick", "Okonkwo", "Salvatierra", "Bellweather", "Ntanda",
              "Krishnamurthy", "Faraday", "Oyelowo"]
AGENTS = ("Analyst-1", "Analyst-2", "Analyst-3", "Analyst-4")

#: Institutions used only by the private fact, so its unique token cannot
#: collide with anything in the shared record. Utterance detection is then a
#: plain string match rather than a judgement call.
PRIVATE_ORGS = ["Halvern Dynamics", "Castleforth Group", "Ridgemoor Labs",
                "Trellis Aeronautics", "Vantage Kiln", "Ashford Meridian",
                "Pellinore Systems", "Quillon Metrics"]

POSITIVES = [
    "shipped the {thing} platform end to end",
    "led a team of {n} through the {thing} migration",
    "published the reference work on {thing}",
    "turned around the failing {thing} programme",
    "holds three patents relating to {thing}",
    "rebuilt {thing} after the outage and cut incidents by half",
]
THINGS = ["telemetry", "settlement", "provisioning", "indexing", "routing",
          "reconciliation", "observability", "failover"]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    candidates: tuple[str, ...]
    shared_facts: tuple[str, ...]
    private_fact: str
    private_token: str
    truth: str
    decoy: str


def _score(facts: list[tuple[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cand, w in facts:
        out[cand] = out.get(cand, 0) + w
    return out


def _argmax_unique(scores: dict[str, int], candidates: tuple[str, ...]) -> str | None:
    """Sole argmax, or None on a tie. A tie is not a hidden profile."""
    best = max(scores.get(c, 0) for c in candidates)
    top = [c for c in candidates if scores.get(c, 0) == best]
    return top[0] if len(top) == 1 else None


def make_scenario(seed: int) -> Scenario | None:
    rng = random.Random(seed)
    cands = rng.sample(CANDIDATES, 3)
    decoy, truth, third = cands[0], cands[1], cands[2]
    org = rng.choice(PRIVATE_ORGS)

    tmpl = rng.sample(POSITIVES, 6)
    things = rng.sample(THINGS, 6)

    def pos(cand: str, i: int) -> tuple[str, int, str]:
        text = tmpl[i].format(thing=things[i], n=rng.choice((6, 9, 12, 20)))
        return cand, 1, f"{cand} {text}."

    # Shared record: decoy 3 positives, truth 2, third 1 -> decoy leads.
    shared = [pos(decoy, 0), pos(decoy, 1), pos(decoy, 2),
              pos(truth, 3), pos(truth, 4), pos(third, 5)]
    # Private: disqualifying for the decoy, heavy enough to reverse the lead.
    private = (decoy, -4,
               f"{decoy} was dismissed from {org} for falsifying test results, "
               f"which is not recorded anywhere in the shared file.")

    shared_scores = _score([(c, w) for c, w, _ in shared])
    full_scores = _score([(c, w) for c, w, _ in shared] + [(private[0], private[1])])

    # ---- the analytic gate ----
    a_shared = _argmax_unique(shared_scores, tuple(cands))
    a_full = _argmax_unique(full_scores, tuple(cands))
    if a_shared is None or a_full is None:
        return None
    if a_shared == truth:          # shared record must mislead
        return None
    if a_full != truth:            # private fact must uniquely correct it
        return None

    texts = [t for _, _, t in shared]
    rng.shuffle(texts)
    return Scenario(
        scenario_id=f"turn-s{seed}",
        candidates=tuple(cands),
        shared_facts=tuple(texts),
        private_fact=private[2],
        private_token=org,
        truth=truth,
        decoy=a_shared,
    )


def system_for(sc: "Scenario") -> str:
    """System prompt, with the candidate count taken from the scenario.

    TURN-1 hardcoded "three candidates"; TURN-2 uses two, and a prompt that
    contradicts its own candidate list is a defect that would have been very
    hard to see in the results.
    """
    word = {2: "two", 3: "three", 4: "four"}.get(len(sc.candidates), str(len(sc.candidates)))
    return SYSTEM.replace("exactly one of three candidates",
                          f"exactly one of {word} candidates")


SYSTEM = (
    "You are one of four analysts on a hiring panel. The panel must recommend "
    "exactly one of three candidates.\n\n"
    "You each have the shared file. Some of you also hold notes that are not in "
    "the shared file. Nobody has been told who holds what.\n\n"
    "Speak in your own voice, in at most three sentences. Say what you think "
    "and why. If you hold information the others have not mentioned, it is your "
    "decision whether and how to raise it. Do not invent facts."
)


def agent_prompt(sc: Scenario, agent: str, is_holder: bool,
                 transcript: list[tuple[str, str]], round_no: int) -> str:
    lines = [f"You are {agent}.", "", "SHARED FILE:"]
    lines += [f"- {f}" for f in sc.shared_facts]
    if is_holder:
        lines += ["", "YOUR PRIVATE NOTE (not in the shared file):",
                  f"- {sc.private_fact}"]
    lines += ["", f"CANDIDATES: {', '.join(sc.candidates)}"]
    if transcript:
        lines += ["", "DISCUSSION SO FAR:"]
        lines += [f"{who}: {msg}" for who, msg in transcript]
    lines += ["", f"Round {round_no}. Give your contribution now (max 3 sentences)."]
    return "\n".join(lines)


def vote_prompt(sc: Scenario, agent: str, is_holder: bool,
                transcript: list[tuple[str, str]]) -> str:
    lines = [f"You are {agent}.", "", "SHARED FILE:"]
    lines += [f"- {f}" for f in sc.shared_facts]
    if is_holder:
        lines += ["", "YOUR PRIVATE NOTE (not in the shared file):",
                  f"- {sc.private_fact}"]
    lines += ["", "FULL DISCUSSION:"]
    lines += [f"{who}: {msg}" for who, msg in transcript]
    lines += ["", f"CANDIDATES: {', '.join(sc.candidates)}",
              "Cast your final vote for exactly one candidate."]
    return "\n".join(lines)


def speak_body(sc: Scenario, agent: str, is_holder: bool,
               transcript: list[tuple[str, str]], round_no: int, model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 300,
        "thinking": {"type": "disabled"},
        "system": [{"type": "text", "text": system_for(sc)}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": agent_prompt(sc, agent, is_holder, transcript, round_no)}
        ]}],
    }


def vote_body(sc: Scenario, agent: str, is_holder: bool,
              transcript: list[tuple[str, str]], model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 200,
        "thinking": {"type": "disabled"},
        "output_config": {"format": {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {
                "vote": {"type": "string", "enum": list(sc.candidates)},
                "confidence": {"type": "number"},
            },
            "required": ["vote", "confidence"],
            "additionalProperties": False,
        }}},
        "system": [{"type": "text", "text": system_for(sc)}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": vote_prompt(sc, agent, is_holder, transcript)}
        ]}],
    }


def speaking_order(holder_position: str) -> tuple[tuple[str, ...], str]:
    """Return (order, holder). Only the holder's position changes between
    conditions; the other three keep their relative order, so the manipulation
    is position rather than a general reshuffle."""
    others = [a for a in AGENTS if a != AGENTS[0]]
    holder = AGENTS[0]
    if holder_position == "first":
        return (holder, *others), holder
    if holder_position == "last":
        return (*others, holder), holder
    raise ValueError(holder_position)


# ─────────────────────────── TURN-2 ───────────────────────────
# Two candidates instead of three. Every TURN-1 failure escaped to the third
# option, so a defensible disagreement with the experimenter's weighting had
# somewhere to land and was scored as a group error. With two, the private fact
# disqualifies the leader and the only alternative is the intended answer.


def make_scenario2(seed: int) -> Scenario | None:
    """Two-candidate hidden profile.

    NOTE: passing this gate is necessary and NOT sufficient. It verifies
    consistency with a reference scorer, which is exactly what TURN-1 verified
    before failing. Construct validity is established separately, by the
    full-information calibration gate in PREREGISTRATION-TURN2.md §3.
    """
    rng = random.Random(seed)
    cands = rng.sample(CANDIDATES, 2)
    decoy, truth = cands[0], cands[1]
    org = rng.choice(PRIVATE_ORGS)

    tmpl = rng.sample(POSITIVES, 5)
    things = rng.sample(THINGS, 5)

    def pos(cand, i):
        text = tmpl[i].format(thing=things[i], n=rng.choice((6, 9, 12, 20)))
        return cand, 1, f"{cand} {text}."

    shared = [pos(decoy, 0), pos(decoy, 1), pos(decoy, 2), pos(truth, 3), pos(truth, 4)]
    private = (decoy, -4,
               f"{decoy} was dismissed from {org} for falsifying test results, "
               f"which is not recorded anywhere in the shared file.")

    sh = _score([(c, w) for c, w, _ in shared])
    full = _score([(c, w) for c, w, _ in shared] + [(private[0], private[1])])
    a_sh = _argmax_unique(sh, tuple(cands))
    a_full = _argmax_unique(full, tuple(cands))
    if a_sh is None or a_full is None or a_sh == truth or a_full != truth:
        return None

    texts = [t for _, _, t in shared]
    rng.shuffle(texts)
    return Scenario(
        scenario_id=f"turn2-s{seed}",
        candidates=tuple(cands),
        shared_facts=tuple(texts),
        private_fact=private[2],
        private_token=org,
        truth=truth,
        decoy=a_sh,
    )


# ── TURN-3 ────────────────────────────────────────────────────────────────
#: TURN-2's private fact carried weight -4 against a shared profile favouring
#: the decoy 3-2, so once uttered it settled the question and the last-speaker
#: arm hit 1.000. Weight -2 leaves the truth ahead by one point instead of
#: three. Fixed in PREREGISTRATION-TURN3.md §2 before any data existed.
TURN3_PRIVATE_WEIGHT = -2

MILD_PRIVATE = [
    "{cand} left {org} after a dispute over how test results were reported, "
    "which is not recorded anywhere in the shared file.",
    "{cand}'s last two projects at {org} were quietly reassigned mid-delivery, "
    "which is not recorded anywhere in the shared file.",
    "{cand} was passed over for tenure at {org} following a review of data "
    "handling, which is not recorded anywhere in the shared file.",
]


def make_scenario3(seed: int) -> Scenario | None:
    """Two-candidate hidden profile with a NON-decisive private fact.

    Same structure as make_scenario2; the private fact is weaker, so the truth
    leads by one rather than three once it is on the table. The gate is
    otherwise identical, and passing it still confers no construct validity --
    that is Gate 2's job.
    """
    rng = random.Random(seed)
    cands = rng.sample(CANDIDATES, 2)
    decoy, truth = cands[0], cands[1]
    org = rng.choice(PRIVATE_ORGS)

    tmpl = rng.sample(POSITIVES, 5)
    things = rng.sample(THINGS, 5)

    def pos(cand, i):
        text = tmpl[i].format(thing=things[i], n=rng.choice((6, 9, 12, 20)))
        return cand, 1, f"{cand} {text}."

    shared = [pos(decoy, 0), pos(decoy, 1), pos(decoy, 2), pos(truth, 3), pos(truth, 4)]
    private = (decoy, TURN3_PRIVATE_WEIGHT,
               rng.choice(MILD_PRIVATE).format(cand=decoy, org=org))

    sh = _score([(c, w) for c, w, _ in shared])
    full = _score([(c, w) for c, w, _ in shared] + [(private[0], private[1])])
    a_sh = _argmax_unique(sh, tuple(cands))
    a_full = _argmax_unique(full, tuple(cands))
    if a_sh is None or a_full is None or a_sh == truth or a_full != truth:
        return None

    texts = [t for _, _, t in shared]
    rng.shuffle(texts)
    return Scenario(
        scenario_id=f"turn3-s{seed}",
        candidates=tuple(cands),
        shared_facts=tuple(texts),
        private_fact=private[2],
        private_token=org,
        truth=truth,
        decoy=a_sh,
    )
