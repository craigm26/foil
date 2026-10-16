"""Bases: the units a carrier search ranks and patches.

FOIL takes no side on what the right unit is. Goodfire's "Neural networks are
already clean" argues a network's own neurons are legible enough to study
directly; Anthropic's feature work, and Qwen's own Qwen-Scope release, study
learned dictionaries instead. An instrument meant to be run inside a lab
should accept either and report which it used. So the carrier search is
written against this interface, and every carrier result names its basis.

    NeuronBasis      one coordinate of the gated MLP's hidden layer (the
                     input to down_proj). The registered primary for MECH-1.
    DictionaryBasis  a sparse dictionary: an SAE (reads and writes one site)
                     or a transcoder (reads mlp_in, writes mlp_out). Loaders
                     for Qwen-Scope checkpoints and a plain .npz format.

A unit's attribution score is Δcoefficient x d P(correct)/d coefficient, with
the gradient for a dictionary feature taken through its decoder row. Patching
a dictionary feature adds (f_src - f_dst) * W_dec[i] to the write site and
leaves the reconstruction error untouched, so an all-features patch with an
identity dictionary equals a neuron patch exactly -- a property the tests use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .whitebox import AddPatch, Patch


class NeuronBasis:
    name = "neurons"
    read_site = write_site = "mlp_hidden"

    def layers(self, n_layers: int) -> list[int]:
        return list(range(n_layers))

    def coeffs(self, layer: int, read_act: np.ndarray) -> np.ndarray:
        return read_act

    def coeff_grads(self, layer: int, grad_write: np.ndarray) -> np.ndarray:
        return grad_write

    def patches(self, layer, positions, units, src_coeffs, dst_coeffs) -> list:
        """Set the chosen neurons to their values in the source run."""
        return [Patch(self.write_site, layer, np.asarray(positions), np.asarray(units),
                      np.asarray(src_coeffs))]


@dataclass
class DictionaryBasis:
    """A sparse dictionary per layer.

    layers[l] holds W_enc [d_in, n], b_enc [n], W_dec [n, d_out], b_dec [d_out]
    and, for JumpReLU, threshold [n].
    """

    name: str
    read_site: str
    write_site: str
    weights: dict = field(repr=False)
    activation: str = "relu"          # "relu" | "topk" | "jumprelu" | "identity"
    k: int | None = None
    subtract_b_dec: bool = False

    def layers(self, n_layers: int) -> list[int]:
        return sorted(l for l in self.weights if l < n_layers)

    def coeffs(self, layer: int, read_act: np.ndarray) -> np.ndarray:
        w = self.weights[layer]
        x = read_act - w["b_dec"] if self.subtract_b_dec else read_act
        pre = x @ w["W_enc"] + w["b_enc"]
        if self.activation == "identity":
            return pre
        if self.activation == "relu":
            return np.maximum(pre, 0.0)
        if self.activation == "jumprelu":
            return np.where(pre > w["threshold"], pre, 0.0)
        if self.activation == "topk":
            r = np.maximum(pre, 0.0)
            if self.k is None or self.k >= r.shape[-1]:
                return r
            idx = np.argpartition(-r, self.k - 1, axis=-1)[..., : self.k]
            out = np.zeros_like(r)
            np.put_along_axis(out, idx, np.take_along_axis(r, idx, axis=-1), axis=-1)
            return out
        raise ValueError(self.activation)

    def coeff_grads(self, layer: int, grad_write: np.ndarray) -> np.ndarray:
        """d metric / d f_i = W_dec[i] . d metric / d write_site."""
        return grad_write @ self.weights[layer]["W_dec"].T

    def patches(self, layer, positions, units, src_coeffs, dst_coeffs) -> list:
        """Add Σ (f_src - f_dst) W_dec[i] at each position. Units sharing a
        position are summed into one delta vector."""
        W = self.weights[layer]["W_dec"]
        positions = np.asarray(positions)
        units = np.asarray(units)
        diff = np.asarray(src_coeffs) - np.asarray(dst_coeffs)
        uniq = np.unique(positions)
        deltas = np.zeros((len(uniq), W.shape[1]), dtype=np.float64)
        for j, p in enumerate(uniq):
            sel = positions == p
            deltas[j] = diff[sel] @ W[units[sel]]
        return [AddPatch(self.write_site, layer, uniq, deltas)]

    # ----------------------------------------------------------- loaders

    @classmethod
    def identity(cls, d: int, layers, site: str = "mlp_hidden") -> "DictionaryBasis":
        """A dictionary whose features are the coordinates themselves. Patching
        it must reproduce a neuron patch exactly; the tests hold it to that."""
        eye = np.eye(d)
        w = {l: {"W_enc": eye, "b_enc": np.zeros(d), "W_dec": eye, "b_dec": np.zeros(d)} for l in layers}
        return cls("identity", site, site, w, activation="identity")

    @classmethod
    def from_npz(cls, path: str | Path, *, read_site: str, write_site: str | None = None,
                 activation: str = "relu", k: int | None = None,
                 subtract_b_dec: bool = False, name: str | None = None) -> "DictionaryBasis":
        """One file, keys `L{n}_W_enc` etc. for each layer n."""
        z = np.load(path)
        layers: dict[int, dict] = {}
        for key in z.files:
            head, _, part = key.partition("_")
            layers.setdefault(int(head[1:]), {})[part] = z[key]
        return cls(name or Path(path).stem, read_site, write_site or read_site, layers,
                   activation, k, subtract_b_dec)

    @classmethod
    def from_qwen_scope(cls, directory: str | Path, layers, *, k: int) -> "DictionaryBasis":
        """Qwen-Scope residual SAEs (`layer{n}.sae.pt`, TopK, hook resid_post).

        Their checkpoints store W_enc as [n, d] and W_dec as [d, n]. A
        dictionary is wider than the space it decomposes (n > d), which fixes
        the orientation without trusting a shape convention: both matrices are
        turned into this module's [d_in, n] / [n, d_out]. The encoder does not
        subtract b_dec, matching the release's own demo. Each layer is ~0.5 GB
        at 1.7B scale, so load only the layers you use.
        """
        import torch
        w: dict[int, dict] = {}
        for l in layers:
            sd = torch.load(Path(directory) / f"layer{l}.sae.pt", map_location="cpu", weights_only=True)
            W_enc = sd["W_enc"].float().numpy()
            W_dec = sd["W_dec"].float().numpy()
            if W_enc.shape[0] > W_enc.shape[1]:
                W_enc = W_enc.T
            if W_dec.shape[0] < W_dec.shape[1]:
                W_dec = W_dec.T
            w[l] = {"W_enc": W_enc, "b_enc": sd["b_enc"].float().numpy(),
                    "W_dec": W_dec, "b_dec": sd["b_dec"].float().numpy()}
        return cls(f"qwen-scope:{Path(directory).name}", "resid_post", "resid_post", w,
                   activation="topk", k=k)
