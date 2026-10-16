"""White-box execution: one forward pass, with knockouts, patches and captures.

THE ARRANGEMENT-FIXED OPERATOR
------------------------------
`knockouts` hides spans in place. For each (keys, q_start) pair, every query
at position >= q_start is forbidden from attending to every key in `keys`, in
every layer and every head. In a decoder whose only cross-position mixing is
softmax attention, that cuts every path by which those tokens could reach
anything after them -- while every token keeps its position, so nothing
downstream is rearranged. It is the counterfactual FOIL's API harness could
not express: information removed, arrangement fixed.

The claim "cuts every path" holds only for pure softmax-attention decoders.
A linear-attention or state-space layer carries tokens forward in a recurrent
state no mask can reach, so such models are refused (`eligibility`). Qwen3.5,
which interleaves Gated DeltaNet layers 3:1 with full attention, is the first
model this rule excludes. The leak gate then checks the claim empirically on
the actual weights: overwrite a hidden span's tokens and nothing after it may
move by more than float noise.

NEURONS
-------
For a gated MLP, down_proj(act(gate(x)) * up(x)), a "neuron" is one coordinate
of the input to down_proj -- the unit Goodfire's essay reads and the unit
`Patch(site="mlp_hidden")` sets. The essay warns that SwiGLU neurons have a
double-negative regime (gate pre-activation and up both negative, product
positive) that reading activations alone hides, so captures can record each
neuron's sign regime as well as its value.

Every forward is run with use_cache=False: knockout masks are built for the
full sequence, and a KV cache would hand later queries keys computed before
the mask existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from .local import LocalPrompt

SITES = ("mlp_hidden", "mlp_in", "mlp_out", "resid_post")

#: Regime labels by (sign of act(gate), sign of up). "--" is the
#: double-negative regime the essay names.
REGIMES = {(True, True): "++", (True, False): "+-", (False, True): "-+", (False, False): "--"}


class Ineligible(RuntimeError):
    """The model's architecture cannot support the registered operators."""


def eligibility(config) -> list[str]:
    """Reasons a model cannot be used. Empty means eligible."""
    cfg = getattr(config, "text_config", None) or config
    reasons = []
    lt = getattr(cfg, "layer_types", None)
    if lt:
        other = sorted({t for t in lt if t not in ("full_attention", "sliding_attention")})
        if other:
            reasons.append(
                f"layer types {other}: non-attention token mixers carry information "
                "in a recurrent state that an attention mask cannot block")
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(cfg, name, 0):
            reasons.append(f"{name}={getattr(cfg, name)}: mixture-of-experts MLPs have no fixed per-token neuron set")
            break
    return reasons


@dataclass
class Patch:
    """Set `units` of `site` at `positions` to `values` (one triple per entry)."""

    site: str
    layer: int
    positions: np.ndarray
    units: np.ndarray
    values: np.ndarray


@dataclass
class AddPatch:
    """Add `deltas[i]` to `site`'s activation at `positions[i]` (dictionary bases)."""

    site: str
    layer: int
    positions: np.ndarray
    deltas: np.ndarray


@dataclass
class Result:
    probs: np.ndarray            # renormalised over the actions
    action_logits: np.ndarray
    on_format: float             # next-token probability mass on the actions
    full_logits: np.ndarray | None = None
    captures: dict = field(default_factory=dict)   # (site, layer) -> [seq, d] float32
    regimes: dict = field(default_factory=dict)    # layer -> [seq, d] int8 code
    grads: dict = field(default_factory=dict)      # (site, layer) -> [seq, d] float32


class WhiteBox:
    """An open-weight decoder, loaded for exact, intervenable forward passes."""

    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu",
                 dtype: str = "float32", model=None, tokenizer=None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # "cuda" means the current GPU; name it, so gates passed on one
            # card cannot certify a study on another.
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype_name = dtype
        td = getattr(torch, dtype)
        if self.device.type == "cuda":
            # float32 must mean float32: Ampere-class GPUs otherwise round
            # matmul inputs to TF32 and the registered precision is gone.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if model is None:
            cfg = AutoConfig.from_pretrained(model_id, revision=revision)
            why = eligibility(cfg)
            if why:
                raise Ineligible(f"{model_id}: " + "; ".join(why))
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, dtype=td, attn_implementation="eager")
            except TypeError:  # transformers < 4.56 spells it torch_dtype
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, torch_dtype=td, attn_implementation="eager")
        else:
            why = eligibility(model.config)
            if why:
                raise Ineligible(f"{model_id}: " + "; ".join(why))
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = model.to(self.device).to(td).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        impl = getattr(self.model.config, "_attn_implementation", None)
        if impl not in (None, "eager"):
            raise Ineligible(f"attention implementation {impl!r}; knockouts are verified on 'eager' only")
        self.layers = self._decoder_layers()
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, "self_attn") or not hasattr(getattr(layer, "mlp", None), "down_proj"):
                raise Ineligible(f"layer {i} lacks self_attn or a gated MLP with down_proj")
        self.n_layers = len(self.layers)
        #: Completed forward passes. Every result file reports compute (§10).
        self.passes = 0
        self.d_mlp = self.layers[0].mlp.down_proj.in_features
        self.d_model = self.layers[0].mlp.down_proj.out_features
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    def _decoder_layers(self):
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise Ineligible("cannot locate model.model.layers")
        return list(layers)

    #: Fields that must match for two records to belong to one measurement.
    IDENTITY_KEYS = ("model", "revision", "dtype", "device", "hardware", "torch", "transformers")

    def identity(self) -> dict:
        import transformers
        return {
            "model": self.model_id,
            "revision": self.resolved_revision,
            "dtype": self.dtype_name,
            "device": str(self.device),
            "hardware": hardware_name(self.device),
            "n_layers": self.n_layers,
            "d_mlp": self.d_mlp,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    # ------------------------------------------------------------------ hooks

    def _ko_hook(self, blocked: torch.Tensor):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            q = kv = blocked.shape[0]
            if mask is None:
                dt = kwargs["hidden_states"].dtype if "hidden_states" in kwargs else self.model.dtype
                causal = torch.ones(q, kv, dtype=torch.bool, device=blocked.device).tril()
                mask = torch.zeros(q, kv, dtype=dt, device=blocked.device).masked_fill(
                    ~causal, torch.finfo(dt).min)[None, None]
            elif mask.dtype == torch.bool:
                dt = self.model.dtype
                mask = torch.zeros(mask.shape, dtype=dt, device=mask.device).masked_fill(
                    ~mask, torch.finfo(dt).min)
            if mask.shape[-2:] != (q, kv):
                raise RuntimeError(f"attention mask shape {tuple(mask.shape)} != ({q}, {kv}); "
                                   "knockouts assume a full, uncached sequence")
            kwargs["attention_mask"] = mask.masked_fill(blocked[None, None], torch.finfo(mask.dtype).min)
            return args, kwargs
        return hook

    @staticmethod
    def _apply(x: torch.Tensor, patches: list) -> torch.Tensor:
        x = x.clone()
        for p in patches:
            pos = torch.as_tensor(np.asarray(p.positions), dtype=torch.long, device=x.device)
            if isinstance(p, Patch):
                units = torch.as_tensor(np.asarray(p.units), dtype=torch.long, device=x.device)
                vals = torch.as_tensor(np.asarray(p.values), dtype=x.dtype, device=x.device)
                x[0, pos, units] = vals
            else:
                deltas = torch.as_tensor(np.asarray(p.deltas), dtype=x.dtype, device=x.device)
                x[0, pos, :] = x[0, pos, :] + deltas
        return x

    # -------------------------------------------------------------------- run

    def run(
        self,
        prompt: LocalPrompt,
        *,
        knockouts: Iterable[tuple[Iterable[int], int]] = (),
        patches: Iterable = (),
        capture: Iterable[str] = (),
        regimes: bool = False,
        grad: bool = False,
        keep_full_logits: bool = False,
        position_ids=None,
    ) -> Result:
        """One exact forward pass. Returns the renormalised action distribution.

        knockouts    -- (keys, q_start) pairs; see the module docstring.
        patches      -- Patch / AddPatch objects, applied before the site is used.
        capture      -- sites to record at every layer and position.
        regimes      -- also record each MLP neuron's SwiGLU sign regime.
        grad         -- record d P(correct) / d activation for captured sites.
        position_ids -- override the rotary positions (the arrangement-only
                        arm). The 2D attention mask is always passed
                        explicitly: transformers infers "packed sequences"
                        from any position jump when it is absent, and would
                        silently cut the context into blocks.
        """
        capture = set(capture)
        unknown = capture - set(SITES)
        if unknown:
            raise ValueError(f"unknown sites {unknown}")
        if grad and not capture:
            raise ValueError("grad=True needs at least one captured site")
        ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=self.device)
        seq = ids.shape[1]
        fwd = {"input_ids": ids, "use_cache": False,
               "attention_mask": torch.ones(1, seq, dtype=torch.long, device=self.device)}
        if position_ids is not None:
            pid = torch.as_tensor(np.asarray(position_ids), dtype=torch.long, device=self.device)
            if pid.shape != (seq,):
                raise ValueError("position_ids must give one position per token")
            fwd["position_ids"] = pid[None]
        by_site: dict[tuple[str, int], list] = {}
        for p in patches:
            if p.site not in SITES:
                raise ValueError(f"unknown patch site {p.site!r}")
            by_site.setdefault((p.site, p.layer), []).append(p)

        handles = []
        caps: dict = {}
        live: dict = {}
        regs: dict = {}
        gate_sign: dict = {}

        def keep(site, li, x):
            if site in capture:
                if grad:
                    x.retain_grad()
                    live[(site, li)] = x
                caps[(site, li)] = x.detach()[0].float().cpu().numpy().copy()

        try:
            ko = list(knockouts)
            if ko:
                blocked = torch.zeros(seq, seq, dtype=torch.bool, device=self.device)
                for keys, q_start in ko:
                    keys = list(keys)
                    if keys:
                        if max(keys) >= q_start:
                            raise ValueError("a knocked-out key must precede every blocked query")
                        blocked[q_start:, keys] = True
                for layer in self.layers:
                    handles.append(layer.self_attn.register_forward_pre_hook(
                        self._ko_hook(blocked), with_kwargs=True))

            for li, layer in enumerate(self.layers):
                mlp = layer.mlp
                if ("mlp_in", li) in by_site or "mlp_in" in capture:
                    def pre_mlp(m, args, li=li):
                        x = args[0]
                        if ("mlp_in", li) in by_site:
                            x = self._apply(x, by_site[("mlp_in", li)])
                        keep("mlp_in", li, x)
                        return (x,) + tuple(args[1:])
                    handles.append(mlp.register_forward_pre_hook(pre_mlp))
                if ("mlp_hidden", li) in by_site or "mlp_hidden" in capture:
                    def pre_down(m, args, li=li):
                        x = args[0]
                        if ("mlp_hidden", li) in by_site:
                            x = self._apply(x, by_site[("mlp_hidden", li)])
                        keep("mlp_hidden", li, x)
                        return (x,) + tuple(args[1:])
                    handles.append(mlp.down_proj.register_forward_pre_hook(pre_down))
                if ("mlp_out", li) in by_site or "mlp_out" in capture:
                    def post_mlp(m, args, out, li=li):
                        x = out
                        if ("mlp_out", li) in by_site:
                            x = self._apply(x, by_site[("mlp_out", li)])
                        keep("mlp_out", li, x)
                        return x
                    handles.append(mlp.register_forward_hook(post_mlp))
                if ("resid_post", li) in by_site or "resid_post" in capture:
                    def post_layer(m, args, out, li=li):
                        h = out[0] if isinstance(out, tuple) else out
                        if ("resid_post", li) in by_site:
                            h = self._apply(h, by_site[("resid_post", li)])
                        keep("resid_post", li, h)
                        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                    handles.append(layer.register_forward_hook(post_layer))
                if regimes and hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                    def g_hook(m, args, out, li=li):
                        gate_sign[li] = (out.detach()[0] > 0).cpu().numpy()
                    def u_hook(m, args, out, li=li):
                        up = (out.detach()[0] > 0).cpu().numpy()
                        g = gate_sign.pop(li)
                        regs[li] = (2 * (~g) + (~up)).astype(np.int8)  # 0 ++, 1 +-, 2 -+, 3 --
                    handles.append(mlp.gate_proj.register_forward_hook(g_hook))
                    handles.append(mlp.up_proj.register_forward_hook(u_hook))
            if grad:
                emb = self.model.get_input_embeddings()
                handles.append(emb.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True)))

            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                if prompt.decision_pos != seq - 1:
                    raise ValueError("the decision position must be the last prompt token")
                try:  # unembedding 150k-vocab rows for every position is ~10% of a pass
                    out = self.model(**fwd, logits_to_keep=1)
                except TypeError:
                    out = self.model(**fwd)
                self.passes += 1
                logits = out.logits[0, -1].float()
                act_ids = torch.as_tensor(prompt.action_ids, dtype=torch.long, device=logits.device)
                a_logits = logits[act_ids]
                probs = torch.softmax(a_logits, dim=-1)
                on_format = torch.softmax(logits, dim=-1)[act_ids].sum()
                grads = {}
                if grad:
                    probs[prompt.correct_index].backward()
                    for key, x in live.items():
                        grads[key] = x.grad[0].float().cpu().numpy().copy()
            return Result(
                probs=probs.detach().cpu().numpy().astype(np.float64),
                action_logits=a_logits.detach().cpu().numpy().astype(np.float64),
                on_format=float(on_format.detach()),
                full_logits=logits.detach().cpu().numpy() if keep_full_logits else None,
                captures=caps,
                regimes=regs,
                grads=grads,
            )
        finally:
            for h in handles:
                h.remove()


def hardware_name(device: torch.device) -> str:
    """The processor the arithmetic ran on: the GPU's name, or the CPU model."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine() or "unknown"


def regime_name(code: int) -> str:
    return ("++", "+-", "-+", "--")[int(code)]
