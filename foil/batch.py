"""Message Batches executor — 50% cost reduction.

A research harness has no latency requirement, so synchronous sampling was
simply the wrong tool: every call so far paid full price for a result nobody
was waiting on. This trades minutes-to-hours of turnaround for half the bill.

Why prompt caching was never the answer here: `cache_control` has been set on
every request since the first commit and has never engaged once. The invariant
prefix is ~171 tokens against a 1024-token minimum on Sonnet-class models and
512 on Opus-class, so the marker is silently inert. Caching only pays when a
large prefix is shared across many calls; in this harness the shared part is
small and the varying part is most of the request. Batching is the lever that
actually applies.

Same zero-dependency posture as `execute.py`: stdlib HTTP only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .execute import ApiError, Ledger, Usage, load_prices

API_VERSION = "2023-06-01"
BASE = "https://api.anthropic.com"

#: Anthropic caps a batch at 100k requests / 256MB. Well under both, but keep
#: batches modest so a single failure costs little and progress is visible.
MAX_PER_BATCH = 2000


class BatchExecutor:
    """Submit many requests, poll once, collect all results.

    Deliberately NOT a drop-in for `Executor`: the batch path cannot be
    interleaved with per-call retry, so the two are kept separate rather than
    hidden behind one interface that behaves differently depending on a flag.
    """

    def __init__(self, store_path: Path, model: str, api_key: str, prices: list[dict]):
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.api_key = api_key
        self.ledger = Ledger(model=model, prices=prices)
        self._cache: dict[str, list[dict]] = {}
        if store_path.exists():
            for line in store_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._cache.setdefault(r["custom_id"], []).append(r)

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "content-type": "application/json",
                "anthropic-version": API_VERSION,
                "x-api-key": self.api_key,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode()[:600]) from None

    def have(self, custom_id: str) -> int:
        return len(self._cache.get(custom_id, []))

    def seed_cache(self, body: dict) -> None:
        """Write the shared prefix to cache before submitting the bulk batch.

        Batch requests run concurrently and in any order, so cache hits are
        best-effort: if the whole batch is submitted at once, many requests race
        the first write and all pay full price. Sending ONE request first and
        waiting for it establishes the entry, after which the rest read it.

        This is a real synchronous request, not a `max_tokens: 0` warm-up --
        that form is rejected when `output_config.format` is set, and rejected
        inside a batch entirely.
        """
        req = urllib.request.Request(
            f"{BASE}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "anthropic-version": API_VERSION, "x-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                u = json.loads(resp.read()).get("usage", {}) or {}
            wrote = u.get("cache_creation_input_tokens", 0) or 0
            read = u.get("cache_read_input_tokens", 0) or 0
            print(f"  cache seed: wrote {wrote} tok, read {read} tok", flush=True)
            if wrote == 0 and read == 0:
                print("  ⚠ prefix did not cache -- below the model minimum, or "
                      "a field before the breakpoint varies", flush=True)
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode()[:600]) from None

    def submit(self, requests: list[tuple[str, dict]]) -> list[str]:
        """requests = [(custom_id, message_body), ...]. Returns batch ids.

        Already-collected custom_ids are skipped, so a re-run after an
        interruption costs nothing for work already paid for.
        """
        pending = [(cid, body) for cid, body in requests if not self.have(cid)]
        if not pending:
            return []
        ids = []
        for i in range(0, len(pending), MAX_PER_BATCH):
            chunk = pending[i : i + MAX_PER_BATCH]
            res = self._req(
                "POST",
                "/v1/messages/batches",
                {"requests": [{"custom_id": c, "params": b} for c, b in chunk]},
            )
            ids.append(res["id"])
            print(f"  batch {res['id']} submitted ({len(chunk)} requests)", flush=True)
        return ids

    def wait(self, batch_ids: list[str], poll_s: int = 30, timeout_s: int = 86400) -> None:
        start = time.time()
        remaining = list(batch_ids)
        while remaining and time.time() - start < timeout_s:
            still = []
            for bid in remaining:
                b = self._req("GET", f"/v1/messages/batches/{bid}")
                if b["processing_status"] == "ended":
                    print(f"  batch {bid} ended: {b['request_counts']}", flush=True)
                else:
                    still.append(bid)
            remaining = still
            if remaining:
                time.sleep(poll_s)
        if remaining:
            raise TimeoutError(f"batches still processing after {timeout_s}s: {remaining}")

    def collect(self, batch_ids: list[str]) -> None:
        """Stream results into the store. Results arrive in ANY order, so they
        are keyed by custom_id and never by position."""
        with self.store_path.open("a") as fh:
            for bid in batch_ids:
                req = urllib.request.Request(
                    f"{BASE}/v1/messages/batches/{bid}/results",
                    headers={
                        "anthropic-version": API_VERSION,
                        "x-api-key": self.api_key,
                    },
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for raw in resp.read().decode().splitlines():
                        if not raw.strip():
                            continue
                        r = json.loads(raw)
                        if r["result"]["type"] != "succeeded":
                            self.ledger.api_failures += 1
                            if len(self.ledger.failure_bodies) < 5:
                                self.ledger.failure_bodies.append(json.dumps(r["result"])[:300])
                            continue
                        msg = r["result"]["message"]
                        u = msg.get("usage", {}) or {}
                        usage = Usage(
                            input_tokens=u.get("input_tokens", 0) or 0,
                            output_tokens=u.get("output_tokens", 0) or 0,
                            cache_read_input_tokens=u.get("cache_read_input_tokens", 0) or 0,
                            cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0) or 0,
                        )
                        self.ledger.record(usage)
                        text = "".join(
                            b.get("text", "") for b in msg.get("content", [])
                            if b.get("type") == "text"
                        )
                        rec = {
                            "custom_id": r["custom_id"],
                            "text": text,
                            "usage": asdict(usage),
                            "model": msg.get("model", self.model),
                            "stop_reason": msg.get("stop_reason"),
                        }
                        fh.write(json.dumps(rec) + "\n")
                        self._cache.setdefault(r["custom_id"], []).append(rec)

    def get(self, custom_id: str) -> list[dict]:
        return self._cache.get(custom_id, [])

    def summary(self) -> dict:
        s = self.ledger.summary()
        # Batch pricing is half of list. The bundled table is list price, so
        # halve it here rather than shipping a second table that could drift.
        if s.get("cost_usd") is not None:
            s["cost_usd_list"] = s["cost_usd"]
            s["cost_usd"] = s["cost_usd"] / 2
            s["pricing_note"] = "Batches API: 50% of list; cost_usd_list is the synchronous equivalent"
        return s
