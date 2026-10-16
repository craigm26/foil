"""White-box invariants on a tiny random Qwen3, built offline.

No downloads: a byte-level BPE is trained on FOIL's own texts in-process and a
two-layer Qwen3 is initialised from a seed. These tests prove the operators do
what the pre-registration says, on the real Qwen3 module code, before any real
weights are loaded. They skip when torch/transformers are absent, so
`./run.sh test` stays green on the dependency-free core.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAVE = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))

CHAT = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)

_CACHE: dict = {}


def tiny():
    """(tokenizer, WhiteBox) -- built once per process."""
    if "wb" in _CACHE:
        return _CACHE["tok"], _CACHE["wb"]
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from foil import env3
    from foil.nulls import paraphrase
    from foil.render import Operator, _ablate
    from mech.whitebox import WhiteBox

    corpus = [env3.SYSTEM_TEXT, env3.WORLD_TEXT, env3.QUERY_TEXT]
    for seed in range(300):
        ep = env3.make_episode_v3(seed)
        if ep is None:
            continue
        for s in ep.reports:
            corpus += [s.text, paraphrase(s).text]
            corpus += [_ablate(s, op).text for op in (Operator.NULL, Operator.NEUTRAL, Operator.INVERT)]
    for r in env3.ROUTES:
        corpus += ['{"route": "' + r + '"'] * 50 + [r] * 50
    corpus += ["<think>\n\n</think>\n\n", "assistant\n", "system\n", "user\n"] * 5

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=700, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tok.chat_template = CHAT

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=512, tie_word_embeddings=True,
                      initializer_range=0.5, attn_implementation="eager")
    wb = WhiteBox("tiny-qwen3", model=Qwen3ForCausalLM(cfg), tokenizer=tok)
    _CACHE.update(tok=tok, wb=wb)
    return tok, wb


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class LocalRender(unittest.TestCase):
    def test_spans_map_cleanly_for_every_arm(self):
        from foil.env3 import make_episode_v3
        from mech.gates import _arms, _span_problems
        tok, _ = tiny()
        for seed in (9000, 9001, 9002):
            for pr in _arms(tok, make_episode_v3(seed), seed):
                self.assertEqual(_span_problems(pr), [], pr.episode_id)

    def test_summary_tokens_are_punctuation(self):
        from foil.env3 import make_episode_v3
        from mech.local import is_boundary_text, render_local
        tok, _ = tiny()
        pr = render_local(tok, make_episode_v3(9000))
        for sm in pr.spans.values():
            for i in sm.summary:
                self.assertTrue(is_boundary_text(pr.text[pr.offsets[i][0]:pr.offsets[i][1]]))
            for i in sm.content:
                self.assertFalse(is_boundary_text(pr.text[pr.offsets[i][0]:pr.offsets[i][1]]))

    def test_readout_is_one_token_per_action(self):
        from foil.env3 import make_episode_v3
        from mech.local import render_local
        tok, _ = tiny()
        pr = render_local(tok, make_episode_v3(9000))
        self.assertEqual(tok.convert_ids_to_tokens(list(pr.action_ids)), list(pr.actions))
        self.assertEqual(pr.decision_pos, len(pr.input_ids) - 1)
        self.assertTrue(pr.text.endswith('{"route": "'))

    def test_claims_round_trip(self):
        from foil.env3 import _draw_claims, make_episode_v3  # noqa: F401
        from mech.local import claims_from_text
        ep = make_episode_v3(9000)
        claims = claims_from_text([s.text for s in ep.reports])
        # Rendering lists clear routes before blocked ones, so compare sets.
        self.assertEqual({k: set(v) for k, v in claims.items()}, {k: set(v) for k, v in ep.coverage.items()})

    def test_honest_variant_pins_truth_when_identifiable(self):
        from foil.env3 import make_episode_v3
        from mech.gates import construct_seeds
        from mech.local import claims_from_text, honest_variant, identified_route
        tok, _ = tiny()
        for seed in construct_seeds(tok, 5):
            ep = make_episode_v3(seed)
            hv = honest_variant(tok, ep)
            lines = [sm.text for sm in hv.spans.values()]
            self.assertEqual(identified_route(claims_from_text(lines), tuple(ep.actions)), ep.correct_action)
            liar = next(s for s, r in ep.reliability.items() if r <= 0.0)
            original = next(s.text for s in ep.reports if s.source_id == liar)
            self.assertNotEqual(hv.spans[liar].text, original)


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class Knockout(unittest.TestCase):
    def setUp(self):
        from foil.env3 import make_episode_v3
        from mech.local import render_local
        self.tok, self.wb = tiny()
        self.pr = render_local(self.tok, make_episode_v3(9001))

    def _scramble(self, keys, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(self.pr.input_ids)
        for k in keys:
            ids[k] = int(rng.integers(3, len(self.tok)))
        return self.pr.with_ids(tuple(ids))

    def test_hidden_span_cannot_leak(self):
        from mech.local import knock
        for s in self.pr.spans:
            keys, q0 = knock(self.pr, s)
            edited = self._scramble(keys)
            a = self.wb.run(self.pr, knockouts=[(keys, q0)], keep_full_logits=True)
            b = self.wb.run(edited, knockouts=[(keys, q0)], keep_full_logits=True)
            self.assertEqual(float(np.max(np.abs(a.full_logits - b.full_logits))), 0.0, s)

    def test_the_leak_test_can_fail(self):
        # Without the knockout, the same overwrite must move the logits;
        # otherwise the test above proves nothing.
        from mech.local import knock
        keys, _ = knock(self.pr, "Alpha")
        a = self.wb.run(self.pr, keep_full_logits=True)
        b = self.wb.run(self._scramble(keys), keep_full_logits=True)
        self.assertGreater(float(np.max(np.abs(a.full_logits - b.full_logits))), 1e-3)

    def test_nothing_at_or_before_the_span_changes(self):
        from mech.local import knock
        keys, q0 = knock(self.pr, "Charlie")
        a = self.wb.run(self.pr, capture={"mlp_hidden", "resid_post"})
        b = self.wb.run(self.pr, knockouts=[(keys, q0)], capture={"mlp_hidden", "resid_post"})
        for key in a.captures:
            np.testing.assert_array_equal(a.captures[key][:q0], b.captures[key][:q0])
        moved = max(float(np.max(np.abs(a.captures[k][q0:] - b.captures[k][q0:]))) for k in a.captures)
        self.assertGreater(moved, 0.0)

    def test_empty_knockout_is_a_no_op(self):
        a = self.wb.run(self.pr, keep_full_logits=True)
        b = self.wb.run(self.pr, knockouts=[((), 5)], keep_full_logits=True)
        np.testing.assert_array_equal(a.full_logits, b.full_logits)

    def test_key_after_query_start_is_refused(self):
        with self.assertRaises(ValueError):
            self.wb.run(self.pr, knockouts=[((10,), 10)])

    def test_partial_knockouts_partition_the_span(self):
        from mech.local import knock
        sk, q1 = knock(self.pr, "Delta", "summary")
        ck, q2 = knock(self.pr, "Delta", "content")
        ak, q3 = knock(self.pr, "Delta", "all")
        ik, _ = knock(self.pr, "Delta", "inner")
        fk, _ = knock(self.pr, "Delta", "final")
        self.assertEqual(q1, q2)
        self.assertEqual(q2, q3)
        self.assertEqual(sorted(sk + ck), sorted(ak))
        self.assertEqual(sorted(ik + fk), sorted(sk))  # the final token is punctuation


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class Positions(unittest.TestCase):
    def setUp(self):
        from foil.env3 import make_episode_v3
        from mech.local import render_local
        self.tok, self.wb = tiny()
        self.pr = render_local(self.tok, make_episode_v3(9005))

    def test_default_positions_are_a_no_op(self):
        a = self.wb.run(self.pr, keep_full_logits=True)
        b = self.wb.run(self.pr, position_ids=list(range(len(self.pr.input_ids))), keep_full_logits=True)
        np.testing.assert_array_equal(a.full_logits, b.full_logits)

    def test_a_position_jump_never_becomes_packed_sequences(self):
        # transformers infers packed sequences from any position jump when no
        # 2D mask is passed, and would silently block-diagonalise attention.
        import torch
        from mech.local import shifted_positions
        seen = {}

        def grab(module, args, kwargs):
            seen["mask"] = kwargs["attention_mask"].detach().clone()  # returns None: observe only

        h = self.wb.layers[0].self_attn.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            for delta in (-3, 1):
                seen.clear()
                self.wb.run(self.pr, position_ids=shifted_positions(self.pr, "Bravo", delta))
                m = seen["mask"][0, 0]
                n = m.shape[0]
                causal = torch.ones(n, n, dtype=torch.bool).tril()
                self.assertTrue(bool(torch.all(m[causal] == 0)), delta)
                self.assertTrue(bool(torch.all(m[~causal] < -1e30)), delta)
        finally:
            h.remove()

    def test_shift_moves_only_later_tokens(self):
        from mech.local import shifted_positions
        q0 = self.pr.spans["Bravo"].last + 1
        pos = shifted_positions(self.pr, "Bravo", -3)
        self.assertEqual(pos[:q0], list(range(q0)))
        self.assertEqual(pos[q0], q0 - 3)


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class PatchesAndGradients(unittest.TestCase):
    def setUp(self):
        from foil.env3 import make_episode_v3
        from mech.local import render_local
        self.tok, self.wb = tiny()
        self.pr = render_local(self.tok, make_episode_v3(9002))

    def test_capture_and_grad_do_not_change_the_answer(self):
        a = self.wb.run(self.pr)
        b = self.wb.run(self.pr, capture={"mlp_hidden"}, grad=True, regimes=True)
        np.testing.assert_allclose(a.probs, b.probs, atol=1e-7)
        self.assertEqual(set(b.grads), {("mlp_hidden", l) for l in range(self.wb.n_layers)})

    def test_self_patch_is_a_no_op(self):
        from mech.whitebox import Patch
        base = self.wb.run(self.pr, capture={"mlp_hidden"})
        act = base.captures[("mlp_hidden", 1)]
        pos = np.array([self.pr.decision_pos] * 5)
        units = np.arange(5)
        r = self.wb.run(self.pr, patches=[Patch("mlp_hidden", 1, pos, units, act[pos, units])])
        np.testing.assert_allclose(r.probs, base.probs, atol=1e-6)

    def test_patch_moves_the_answer(self):
        from mech.whitebox import Patch
        base = self.wb.run(self.pr)
        pos = np.array([self.pr.decision_pos] * 64)
        r = self.wb.run(self.pr, patches=[Patch("mlp_hidden", 2, pos, np.arange(64), np.full(64, 25.0))])
        self.assertGreater(float(np.max(np.abs(r.probs - base.probs))), 1e-3)

    def test_gradient_predicts_a_small_patch(self):
        # Attribution patching's linear estimate must match a real patch for a
        # small perturbation, or carrier rankings would be noise.
        from mech.whitebox import Patch
        c = self.pr.correct_index
        g = self.wb.run(self.pr, capture={"mlp_hidden"}, grad=True)
        layer, pos = 1, self.pr.decision_pos - 3
        grad = g.grads[("mlp_hidden", layer)][pos]
        u = int(np.argmax(np.abs(grad)))
        eps = 1e-3
        val = g.captures[("mlp_hidden", layer)][pos, u] + eps
        r = self.wb.run(self.pr, patches=[Patch("mlp_hidden", layer, np.array([pos]), np.array([u]), np.array([val]))])
        actual = float(r.probs[c] - g.probs[c])
        predicted = float(grad[u] * eps)
        self.assertAlmostEqual(actual, predicted, delta=0.05 * abs(predicted) + 1e-7)

    def test_identity_dictionary_equals_neurons(self):
        from mech.basis import DictionaryBasis, NeuronBasis
        from mech.local import knock
        keys, q0 = knock(self.pr, "Bravo")
        full = self.wb.run(self.pr, capture={"mlp_hidden"})
        ko = self.wb.run(self.pr, knockouts=[(keys, q0)], capture={"mlp_hidden"})
        layer = 1
        pos = np.repeat(np.arange(q0, self.pr.decision_pos + 1), 3)
        units = np.tile(np.arange(3), len(pos) // 3)
        src = ko.captures[("mlp_hidden", layer)][pos, units]
        dst = full.captures[("mlp_hidden", layer)][pos, units]
        neu = NeuronBasis().patches(layer, pos, units, src, dst)
        dic = DictionaryBasis.identity(self.wb.d_mlp, [layer]).patches(layer, pos, units, src, dst)
        a = self.wb.run(self.pr, patches=neu)
        b = self.wb.run(self.pr, patches=dic)
        np.testing.assert_allclose(a.probs, b.probs, atol=1e-6)

    def test_topk_dictionary_activation(self):
        from mech.basis import DictionaryBasis
        rng = np.random.default_rng(0)
        d, n = 8, 32
        w = {0: {"W_enc": rng.normal(size=(d, n)), "b_enc": np.zeros(n),
                 "W_dec": rng.normal(size=(n, d)), "b_dec": np.zeros(d)}}
        basis = DictionaryBasis("t", "resid_post", "resid_post", w, activation="topk", k=4)
        f = basis.coeffs(0, rng.normal(size=(5, d)))
        self.assertTrue(np.all((f > 0).sum(axis=-1) <= 4))
        self.assertTrue(np.all(f >= 0))


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class Eligibility(unittest.TestCase):
    def test_refuses_recurrent_mixers_and_experts(self):
        from types import SimpleNamespace
        from mech.whitebox import eligibility
        hybrid = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
        self.assertTrue(eligibility(hybrid))
        moe = SimpleNamespace(layer_types=["full_attention"], num_experts=64)
        self.assertTrue(eligibility(moe))
        dense = SimpleNamespace(layer_types=["full_attention", "sliding_attention"])
        self.assertEqual(eligibility(dense), [])
        wrapped = SimpleNamespace(text_config=hybrid)
        self.assertTrue(eligibility(wrapped))


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class StudyLoop(unittest.TestCase):
    def test_episode_record_and_identity(self):
        from foil.env3 import make_episode_v3
        from mech import analysis as A
        from mech.study import measure_episode
        _, wb = tiny()
        rec = measure_episode(wb, make_episode_v3(9003), 9003)
        self.assertEqual(len(rec["n1"]), 6)
        self.assertEqual(len(rec["units"]), 4)
        self.assertEqual(sorted(rec["ref"]), ["Alpha", "Bravo", "Charlie", "Delta"])
        src = ["Alpha", "Bravo", "Charlie", "Delta"]
        r = 9003 % 4
        self.assertEqual(rec["arms_order"], src[r:] + src[:r])
        for u in rec["units"]:
            c = A.components(u)  # raises if the decomposition identity fails
            self.assertAlmostEqual(c["info"] + c["arr_null"], c["info_rev"] + c["arr_rev"])
            self.assertIn("ko_final", u["p"])
            self.assertEqual(u["slot"], rec["arms_order"].index(u["source"]))
        self.assertTrue(rec["consistency"]["ok"])
        self.assertGreater(rec["r0_min_on_format"], 0.0)

    def test_carriers_run_and_report(self):
        from foil.env3 import make_episode_v3
        from mech.basis import NeuronBasis
        from mech.local import knock, render_local
        from mech.study import find_carriers
        tok, wb = tiny()
        pr = render_local(tok, make_episode_v3(9004))
        full = wb.run(pr, capture={"mlp_hidden"}, grad=True, regimes=True)
        keys, q0 = knock(pr, "Alpha")
        ko = wb.run(pr, knockouts=[(keys, q0)], capture={"mlp_hidden"}, regimes=True)
        info = float(ko.probs[pr.correct_index] - full.probs[pr.correct_index]) or -0.2
        car = find_carriers(wb, wb.run, pr, "Alpha", full, ko, info, NeuronBasis())
        again = find_carriers(wb, wb.run, pr, "Alpha", full, ko, info, NeuronBasis())
        self.assertEqual(car, again)  # fully deterministic
        self.assertEqual(sorted(car["by_k"]), [4, 16, 64, 256])
        for key in ("matched", "top_abs"):
            self.assertIsInstance(car[key], float)
        for c in car["top"]:
            self.assertGreater(c["position"], pr.spans["Alpha"].last)
            self.assertIn(c["regime_full"], {"++", "+-", "-+", "--"})
        pre = car["pre_answer"]
        for c in pre["top"]:
            self.assertNotEqual(pr.region[c["position"]], "answer")
            self.assertLess(c["position"], pr.answer_start)

    def test_identity_names_device_and_hardware(self):
        _, wb = tiny()
        ident = wb.identity()
        self.assertEqual(ident["device"], "cpu")
        self.assertTrue(ident["hardware"])
        self.assertIn("hardware", type(wb).IDENTITY_KEYS)

    def test_resume_refuses_other_code_and_duplicates(self):
        from mech.study import run_study
        _, wb = tiny()
        ident = wb.identity()
        row = {"episode": "v3-s9000", "identity": ident, "consistency": {"ok": True},
               "provenance": {"git_commit": "old", "git_dirty": False}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(RuntimeError):
                run_study(wb, [9000], path, log=lambda *_: None,
                          provenance={"git_commit": "new", "git_dirty": False})
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(RuntimeError):
                run_study(wb, [9000], path, log=lambda *_: None,
                          provenance={"git_commit": "old", "git_dirty": False})

    def test_resume_refuses_a_void_rung(self):
        from mech.study import run_study
        _, wb = tiny()
        ident = wb.identity()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.jsonl"
            path.write_text(json.dumps({"episode": "v3-s9000", "identity": ident,
                                        "consistency": {"ok": False}}) + "\n")
            with self.assertRaises(RuntimeError):
                run_study(wb, [9000, 9001], path, log=lambda *_: None)

    def test_resume_refuses_a_different_model(self):
        from mech.study import run_study
        _, wb = tiny()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.jsonl"
            path.write_text(json.dumps({"episode": "v3-s9000", "identity": {
                "model": "other", "revision": None, "dtype": "float32"}}) + "\n")
            with self.assertRaises(RuntimeError):
                run_study(wb, [9000], path, log=lambda *_: None)


@unittest.skipUnless(HAVE, "torch/transformers not installed")
class Gates(unittest.TestCase):
    def test_structural_passes_on_clean_rendering(self):
        from mech.gates import structural_gate
        tok, _ = tiny()
        rep = structural_gate(tok, study_seeds=(9000, 9001, 9002))
        self.assertEqual(rep["outcome"], "PASS", rep["failures"])

    def test_leak_gate_passes_on_a_correct_operator(self):
        from mech.gates import leak_gate
        _, wb = tiny()
        rep = leak_gate(wb, seeds=(9600,))
        self.assertEqual(rep["outcome"], "PASS", rep)
        self.assertEqual(rep["max_abs_logit_diff_with_knockout"], 0.0)


if __name__ == "__main__":
    unittest.main()
