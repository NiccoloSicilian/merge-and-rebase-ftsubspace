"""Checks for subspace-constrained fine-tuning on text models.

No transformers model is downloaded: a module reproducing the T5 parameter
layout is enough to test what the text path actually decides, which is which
tensors it touches and what its reference probe differentiates.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from merge_and_rebase.finetune import subspace_text as st


WIDTH, FF, VOCAB, BLOCKS, LABELS = 32, 64, 128, 2, 3


class _Attention(nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(WIDTH, WIDTH, bias=False))
        if bias:
            self.relative_attention_bias = nn.Embedding(32, 4)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.DenseReluDense = nn.Module()
        self.DenseReluDense.wi = nn.Linear(WIDTH, FF, bias=False)
        self.DenseReluDense.wo = nn.Linear(FF, WIDTH, bias=False)


class _Block(nn.Module):
    def __init__(self, first: bool):
        super().__init__()
        self.layer = nn.ModuleList()
        attn = nn.Module()
        attn.SelfAttention = _Attention(bias=first)
        self.layer.append(attn)
        self.layer.append(_FeedForward())


class T5Stub(nn.Module):
    """The t5 parameter layout, plus a sequence-classification head."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(VOCAB, WIDTH)
        self.encoder = nn.Module()
        self.encoder.embed_tokens = self.shared
        self.encoder.block = nn.ModuleList(_Block(i == 0) for i in range(BLOCKS))
        self.encoder.final_layer_norm = nn.LayerNorm(WIDTH)
        self.classification_head = nn.Module()
        self.classification_head.dense = nn.Linear(WIDTH, WIDTH)
        self.classification_head.out_proj = nn.Linear(WIDTH, LABELS)


class EligibilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = T5Stub()

    def test_only_attention_and_feedforward_matrices(self):
        names = {n for n, _ in st.eligible_matrices(self.model)}
        self.assertEqual(len(names), BLOCKS * 4 + BLOCKS * 2)
        self.assertTrue(all(".SelfAttention." in n or ".DenseReluDense." in n for n in names))

    def test_the_vocabulary_embedding_is_excluded(self):
        names = {n for n, _ in st.eligible_matrices(self.model)}
        # 128x32 and tied to the output head: shared by every expert.
        self.assertNotIn("shared.weight", names)
        self.assertNotIn("encoder.embed_tokens.weight", names)

    def test_the_relative_attention_bias_is_excluded(self):
        names = {n for n, _ in st.eligible_matrices(self.model)}
        self.assertFalse(any("relative_attention_bias" in n for n in names))

    def test_the_head_is_excluded_from_the_subspace(self):
        names = {n for n, _ in st.eligible_matrices(self.model)}
        self.assertFalse(any("classification_head" in n for n in names))

    def test_the_head_is_reported_separately(self):
        head = {n for n, _ in st.head_parameters(self.model)}
        self.assertEqual(
            head,
            {
                "classification_head.dense.weight",
                "classification_head.dense.bias",
                "classification_head.out_proj.weight",
                "classification_head.out_proj.bias",
            },
        )

    def test_an_embedding_is_excluded_even_when_its_name_says_nothing(self):
        # Name patterns are a convention; module type is exact.
        model = nn.Module()
        model.weird = nn.Embedding(VOCAB, WIDTH)
        with self.assertRaisesRegex(RuntimeError, "zero matrices"):
            st.eligible_matrices(model)

    def test_an_architecture_with_no_matrices_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "zero matrices"):
            st.eligible_matrices(nn.LayerNorm(4))


class _Outputs:
    def __init__(self, hidden, loss=None, logits=None):
        self.hidden_states = hidden
        self.loss = loss
        self.logits = logits


class ProbeTest(unittest.TestCase):
    """The reference probe runs on hidden states, not on three logits."""

    def _model(self, hidden):
        class _M(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None,
                        output_hidden_states=False):
                del input_ids, labels
                return _Outputs([hidden], logits=torch.zeros(hidden.shape[0], LABELS))

        return _M()

    def test_padding_is_masked_out(self):
        torch.manual_seed(0)
        hidden = torch.zeros(1, 3, 4)
        hidden[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hidden[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hidden[0, 2] = torch.tensor([0.0, 9.0, 0.0, 0.0])  # padding, must not count
        hidden.requires_grad_(True)
        batch = {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
        generator = torch.Generator().manual_seed(0)
        value = st.reference_objective(self._model(hidden), batch, torch.device("cpu"), generator)
        value.backward()
        # The masked position receives no gradient.
        self.assertAlmostEqual(float(hidden.grad[0, 2].abs().sum()), 0.0, places=6)
        self.assertGreater(float(hidden.grad[0, 0].abs().sum()), 0.0)

    def test_the_decoder_representation_wins_on_an_encoder_decoder(self):
        # The encoder output does not depend on the decoder weights, so probing
        # it would leave every decoder matrix without a denominator.
        enc = torch.randn(1, 5, 4)
        dec = torch.randn(1, 1, 4)
        outputs = type(
            "Seq2SeqOut", (),
            {"decoder_hidden_states": [dec], "encoder_hidden_states": [enc],
             "hidden_states": None, "encoder_last_hidden_state": enc},
        )()
        chosen, source = st._last_hidden_state(outputs)
        self.assertIs(chosen, dec)
        self.assertEqual(source, "decoder_hidden_states")

    def test_an_encoder_only_model_uses_its_hidden_states(self):
        hidden = torch.randn(1, 3, 4)
        outputs = type("Out", (), {"hidden_states": [hidden]})()
        chosen, source = st._last_hidden_state(outputs)
        self.assertIs(chosen, hidden)
        self.assertEqual(source, "hidden_states")

    def test_a_bare_last_hidden_state_is_accepted(self):
        hidden = torch.randn(1, 3, 4)
        outputs = type("Out", (), {"last_hidden_state": hidden})()
        chosen, source = st._last_hidden_state(outputs)
        self.assertIs(chosen, hidden)
        self.assertEqual(source, "last_hidden_state")

    def test_the_input_mask_is_ignored_when_lengths_disagree(self):
        # The decoder length is unrelated to the input length; applying the
        # encoder mask to it would be meaningless rather than merely wasteful.
        dec = torch.ones(1, 1, 4, requires_grad=True)

        class _M(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None,
                        output_hidden_states=False):
                del input_ids, attention_mask, labels, output_hidden_states
                return type("Out", (), {"decoder_hidden_states": [dec],
                                        "hidden_states": None})()

        batch = {
            "input_ids": torch.zeros(1, 7, dtype=torch.long),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
        }
        value = st.reference_objective(_M(), batch, torch.device("cpu"),
                                       torch.Generator().manual_seed(0))
        value.backward()
        self.assertGreater(float(dec.grad.abs().sum()), 0.0)

    def test_a_model_without_hidden_states_is_reported(self):
        class _M(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None,
                        output_hidden_states=False):
                del input_ids, attention_mask, labels, output_hidden_states
                return _Outputs(None)

        batch = {"input_ids": torch.zeros(1, 2, dtype=torch.long)}
        with self.assertRaisesRegex(RuntimeError, "no hidden states"):
            st.reference_objective(_M(), batch, torch.device("cpu"))

    def test_the_blind_matrix_guard_is_present(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/subspace_text.py"
        ).read_text()
        self.assertIn("they have no denominator", source)
        self.assertIn("The probe must depend on every matrix", source)

    def test_the_task_objective_uses_the_models_own_loss(self):
        sentinel = torch.tensor(1.25, requires_grad=True)

        class _M(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None):
                del input_ids, attention_mask, labels
                return _Outputs(None, loss=sentinel)

        batch = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        self.assertIs(st.task_objective(_M(), batch, torch.device("cpu")), sentinel)

    def test_the_task_objective_falls_back_to_cross_entropy(self):
        class _M(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None):
                del input_ids, attention_mask, labels
                return _Outputs(None, loss=None, logits=torch.zeros(2, LABELS))

        batch = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        value = st.task_objective(_M(), batch, torch.device("cpu"))
        self.assertAlmostEqual(float(value), float(torch.log(torch.tensor(float(LABELS)))), places=5)


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        values = st.validate(None)
        self.assertEqual(values["k"], 64)
        self.assertEqual(values["reference_dataset"], "wikitext")
        self.assertFalse(values["no_reference"])

    def test_misspelled_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            st.validate({"rnak": 8})

    def test_out_of_range_values_are_rejected(self):
        for bad in ({"k": 0}, {"delta": -1.0}, {"batches": 0}):
            with self.assertRaises(ValueError):
                st.validate(bad)

    def test_overrides_are_kept(self):
        values = st.validate({"k": 16, "no_reference": True, "max_length": 64})
        self.assertEqual(values["k"], 16)
        self.assertTrue(values["no_reference"])
        self.assertEqual(values["max_length"], 64)


class BranchWiringTest(unittest.TestCase):
    """Read the file rather than import it.

    Importing train_text pulls the whole data stack, which is not needed to
    assert that the branch is wired and keeps this check runnable wherever the
    source is.
    """

    @property
    def source(self) -> str:
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/train_text.py"
        )
        return path.read_text()

    def test_train_text_routes_the_name(self):
        self.assertIn('elif name == "subspace"', self.source)
        self.assertIn("prepare_text_subspace", self.source)
        self.assertIn("full, linear_probe, peft_lora, subspace", self.source)

    def test_the_loader_and_tokenizer_are_passed(self):
        source = self.source
        self.assertIn("train_loader=train_loader.loader", source)
        self.assertIn("model_tag=_safe_model_tag(build_cfg.model_name_or_path)", source)

    def test_the_other_strategies_are_untouched(self):
        source = self.source
        self.assertIn('if name == "full"', source)
        self.assertIn('elif name == "linear_probe"', source)
        self.assertIn('elif name == "peft_lora"', source)


if __name__ == "__main__":
    unittest.main()


def _train_text_importable() -> bool:
    """train_text pulls the whole data stack; the source-level checks do not."""

    try:
        import merge_and_rebase.finetune.train_text  # noqa: F401
    except Exception:
        return False
    return True


class SavingTest(unittest.TestCase):
    """A reparameterised model must be written as the forward pass sees it."""

    @unittest.skipUnless(_train_text_importable(), "train_text needs the data stack")
    def test_the_getter_is_preferred_when_present(self):
        from merge_and_rebase.finetune.train_text import _materialized_state_dict

        model = nn.Linear(3, 2)
        folded = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        model._materialized_state_dict = lambda: folded
        state = _materialized_state_dict(model)
        self.assertTrue(torch.equal(state["weight"], torch.ones(2, 3)))
        self.assertFalse(torch.equal(state["weight"], model.weight.detach()))

    @unittest.skipUnless(_train_text_importable(), "train_text needs the data stack")
    def test_a_plain_model_falls_through(self):
        from merge_and_rebase.finetune.train_text import _materialized_state_dict

        model = nn.Linear(3, 2)
        state = _materialized_state_dict(model)
        self.assertEqual(set(state), {"weight", "bias"})
        self.assertTrue(torch.equal(state["weight"], model.weight.detach()))

    def test_the_save_path_calls_it(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/train_text.py"
        ).read_text()
        # Writing model.state_dict() directly would store the frozen base
        # weights and lose every delta the subspace produced.
        self.assertIn('payload["state_dict"] = _materialized_state_dict(model)', source)

    def test_both_strategy_gates_accept_the_name(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/train_text.py"
        ).read_text()
        # One gate validates the config, the other dispatches. Missing either
        # one fails the run late or lets an unknown name through.
        self.assertIn('{"full", "linear_probe", "peft_lora", "subspace"}', source)
        self.assertIn('elif name == "subspace"', source)


class CanonicalDatasetPathTest(unittest.TestCase):
    """Recent `datasets` rejects bare repository ids."""

    def test_a_bare_name_gains_its_namespace(self):
        from merge_and_rebase.data.text_loaders import canonical_dataset_paths

        # The short form stays first so an older `datasets` keeps working.
        self.assertEqual(canonical_dataset_paths("glue"), ("glue", "nyu-mll/glue"))
        self.assertEqual(canonical_dataset_paths("snli"), ("snli", "stanfordnlp/snli"))
        self.assertEqual(
            canonical_dataset_paths("wikitext"), ("wikitext", "Salesforce/wikitext")
        )

    def test_an_already_namespaced_path_is_left_alone(self):
        from merge_and_rebase.data.text_loaders import canonical_dataset_paths

        for path in ("allenai/scitail", "yangwang825/sick", "my-org/my-data"):
            self.assertEqual(canonical_dataset_paths(path), (path,))

    def test_an_unknown_bare_name_is_returned_unchanged(self):
        from merge_and_rebase.data.text_loaders import canonical_dataset_paths

        self.assertEqual(canonical_dataset_paths("not_a_known_dataset"), ("not_a_known_dataset",))

    def test_every_nli_task_resolves_to_a_namespaced_candidate(self):
        from merge_and_rebase.data.text_loaders import (
            _TASK_SPECS,
            NLI_TASKS,
            canonical_dataset_paths,
        )

        for task in NLI_TASKS:
            candidates = canonical_dataset_paths(_TASK_SPECS[task].hf_path)
            self.assertTrue(
                any("/" in candidate for candidate in candidates),
                f"{task} has no namespaced candidate: {candidates}",
            )

    def test_the_loader_tries_each_candidate(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/data/text_loaders.py"
        ).read_text()
        self.assertIn("for hf_path in canonical_dataset_paths(spec.hf_path):", source)


class SampleCountTest(unittest.TestCase):
    """A Gram estimated from fewer samples than its width is set by the ridge."""

    def test_the_default_covers_the_width_at_the_text_batch_size(self):
        # The text presets use batch 4 with gradient accumulation, so the
        # vision default of 30 batches would be 120 samples for a 768x768 Gram.
        values = st.validate(None)
        samples = values["batches"] * 4
        self.assertGreaterEqual(samples, 768)
        # Clearing the rank threshold is not enough: match vision's five
        # samples per dimension, since the denominator selects on the small
        # eigenvalues and those are the noisiest.
        self.assertGreaterEqual(samples / 768, 5.0)

    def test_the_warning_exists_and_names_the_remedy(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/subspace_text.py"
        ).read_text()
        self.assertIn("samples = values[\"batches\"] * batch_size", source)
        self.assertIn("of directions are set by the ridge", source)

    def test_rank_arithmetic(self):
        # Each batch contributes a gradient of rank at most its batch size.
        for batches, batch, width, covered in [
            (30, 128, 768, True),    # vision
            (30, 4, 768, False),     # the text default before the fix
            (400, 4, 768, True),     # after
        ]:
            self.assertEqual(batches * batch >= width, covered,
                             f"{batches}x{batch} vs {width}")
