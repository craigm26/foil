"""Sampling executor, idempotent sample store, and cost ledger.

Implements PREREGISTRATION.md §6.1 components 4 and 7.

Zero third-party dependencies by design: the harness has to be runnable
unchanged inside a lab against models the author cannot reach (§11), and every
dependency is a reason someone does not run it.

Routing through `mcp-tape llm --port <p>` (set base_url) gives request/response
capture, normalized usage, and TTFT for free. The proxy never retries, so retry
policy stays owned here, where sampling integrity is decided.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .render import request_hash

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"

#: Anthropic will not cache a prefix below this many tokens. A FOIL episode's
#: invariant prefix is small, so caching may simply never engage -- the ledger
#: reports measured cache tokens instead of assuming a saving.
MIN_CACHEABLE_TOKENS = {"haiku": 2048}
DEFAULT_MIN_CACHEABLE = 1024


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """cache_read / (input + cache_read + cache_creation).

        Formula and absence discipline follow mcp-replay's insights-model.js:
        every metric is a finite number or None, never NaN.
        """
        denom = self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens
        if denom == 0:
            return None
        return self.cache_read_input_tokens / denom


def load_prices(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def resolve_price(model: str, table: list[dict]) -> dict | None:
    """Longest-prefix match over the model id.

    Fails CLOSED: an unknown model resolves to no price, never a guessed one.
    This is deliberate -- a silently wrong cost number is worse than a missing
    one when cost is a gate criterion (§8 G4).
    """
    best = None
    for row in table:
        m = row.get("match", "")
        if model.startswith(m) and (best is None or len(m) > len(best["match"])):
            best = row
    return best


def cost_usd(usage: Usage, model: str, table: list[dict]) -> float | None:
    row = resolve_price(model, table)
    if row is None:
        return None
    def rate(k: str) -> float:
        v = row.get(k)
        return float(v) if isinstance(v, (int, float)) else 0.0
    return (
        usage.input_tokens * rate("inPerMtok")
        + usage.output_tokens * rate("outPerMtok")
        + usage.cache_read_input_tokens * rate("cacheReadPerMtok")
        + usage.cache_creation_input_tokens * rate("cacheWritePerMtok")
    ) / 1_000_000


@dataclass
class Ledger:
    """First-class cost accounting (§6.1 component 7), not a log line."""

    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    cache_hits: int = 0
    unparseable: int = 0
    api_failures: int = 0
    failure_bodies: list[str] = field(default_factory=list)
    model: str = ""
    prices: list[dict] = field(default_factory=list)

    def record(self, u: Usage) -> None:
        self.usage.add(u)
        self.calls += 1
        if u.cache_read_input_tokens > 0:
            self.cache_hits += 1

    def summary(self) -> dict:
        c = cost_usd(self.usage, self.model, self.prices)
        return {
            "calls": self.calls,
            "usage": asdict(self.usage),
            "cache_hit_rate": self.usage.cache_hit_rate,
            "calls_with_cache_read": self.cache_hits,
            "unparseable_samples": self.unparseable,
            "api_failures": self.api_failures,
            "failure_bodies": self.failure_bodies,
            "cost_usd": c,
            "cost_note": None if c is not None else f"no price row for {self.model!r}; add one or pass --price-file",
        }


class BudgetExceeded(RuntimeError):
    pass


class ApiError(RuntimeError):
    """An API rejection that carries its response body.

    A bare status code is not a diagnosis; every 400 in this project so far
    has been explained only by the body text.
    """

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class Executor:
    """Idempotent sampler. A request_hash already holding >= n samples is
    never re-executed (§4.2)."""

    def __init__(
        self,
        store_path: Path,
        model: str,
        prices: list[dict],
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        concurrency: int = 8,
    ):
        self.concurrency = concurrency
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.ledger = Ledger(model=model, prices=prices)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_output_tokens = max_output_tokens
        self._cache: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        for line in self.store_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            self._cache.setdefault(rec["request_hash"], []).append(rec)

    def have(self, h: str) -> int:
        return len(self._cache.get(h, []))

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": API_VERSION,
                "x-api-key": self.api_key or "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # The response body carries the only useful diagnosis. Discarding
            # it turns every API rejection into an uninformative status code.
            try:
                detail = e.read().decode()[:600]
            except Exception:
                detail = "<body unreadable>"
            raise ApiError(e.code, detail) from None

    def sample(self, body: dict, n: int, dry_run: bool = False) -> list[dict]:
        """Draw n samples for one fork. Returns the full record list."""
        h = request_hash(body)
        have = self.have(h)
        if dry_run:
            return [{"request_hash": h, "dry_run": True, "needed": max(0, n - have)}]
        if have >= n:
            return self._cache[h][:n]

        if not self.api_key:
            raise RuntimeError(
                "no ANTHROPIC_API_KEY. FOIL makes metered Messages API calls; a Claude Code "
                "subscription cannot serve them. Provision a key scoped to this project."
            )

        if self.max_output_tokens is not None and self.ledger.usage.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(
                f"output-token budget {self.max_output_tokens} reached after {self.ledger.calls} calls"
            )

        out = list(self._cache.get(h, []))
        # HTTP calls run concurrently; the store stays single-writer. Sampling
        # integrity depends on FOIL owning retries, so the backoff lives here
        # rather than in any proxy the requests are routed through.
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._post_with_retry, body) for _ in range(n - have)]
            with self.store_path.open("a") as fh:
                for fut in as_completed(futures):
                    try:
                        resp = fut.result()
                    except ApiError as e:
                        # Never silently drop. A failed draw shrinks the arm,
                        # and an arm that did not reach n samples must be
                        # visible to the analysis rather than assumed full.
                        self.ledger.api_failures += 1
                        if len(self.ledger.failure_bodies) < 5:
                            self.ledger.failure_bodies.append(f"HTTP {e.status}: {e.body[:300]}")
                        continue
                    u = resp.get("usage", {}) or {}
                    usage = Usage(
                        input_tokens=u.get("input_tokens", 0) or 0,
                        output_tokens=u.get("output_tokens", 0) or 0,
                        cache_read_input_tokens=u.get("cache_read_input_tokens", 0) or 0,
                        cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0) or 0,
                    )
                    self.ledger.record(usage)
                    text = "".join(
                        b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"
                    )
                    rec = {
                        "request_hash": h,
                        "text": text,
                        "usage": asdict(usage),
                        "model": resp.get("model", self.model),
                        "stop_reason": resp.get("stop_reason"),
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    out.append(rec)
        self._cache[h] = out
        return out

    def _post_with_retry(self, body: dict) -> dict:
        for attempt in range(5):
            try:
                return self._post(body)
            except ApiError as e:
                # 400 is included deliberately: it is observed intermittently
                # here on requests that succeed when replayed individually, so
                # treating it as permanent kills a run that would have
                # completed. It is retried fewer times than a 429 so a genuine
                # malformed request still surfaces quickly.
                retryable = e.status in (429, 500, 502, 503, 529) or (
                    e.status == 400 and attempt < 2
                )
                if retryable and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")
