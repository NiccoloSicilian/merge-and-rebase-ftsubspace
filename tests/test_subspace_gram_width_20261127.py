"""Gram width and head warm-up.

Two changes with one thing in common: both are about what `A_task` measures
rather than how precisely it is measured.

`micro_batches` widens each Gram term without widening the batch. The loss is a
sum over samples and gradients add, so this must be *exactly* the Gram of one
wide batch -- that exactness is what makes it a faithful substitute for
vision's batch of 128, so it is tested as an identity and not as a tolerance.

`warm_up_head` trains the classification head alone first. The task gradient
reaches the backbone through that head, and at random initialisation it reports
how an arbitrary linear readout propagates instead of what the task needs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from merge_and_rebase.finetune import subspace_bases as sb  # noqa: E402


class _Toy(nn.Module):
    """Two matrices and a head, shaped like the text path: the task loss reaches
    the matrices only through the head."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(6, 8, bias=False)
        self.b = nn.Linear(8, 6, bias=False)
        self.classification_head = nn.Linear(6, 3, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.b(torch.relu(self.a(input_ids.float())))
        logits = self.classification_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return type("Out", (), {"loss": loss, "logits": logits, "last_hidden_state": hidden})()


def _batches(count: int, size: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randn(size, 6, generator=generator),
            "attention_mask": torch.ones(size, 6),
            "labels": torch.randint(0, 3, (size,), generator=generator),
        }
        for _ in range(count)
    ]


def _objective(model, batch, device, generator=None):
    del device, generator
    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
    # Sum, not mean: a wide batch and its micro-batches must weigh each sample
    # identically, and a per-batch mean would rescale by the batch size.
    return outputs.loss * batch["labels"].numel()


class MicroBatchesEqualOneWideBatch(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = _Toy()
        self.matrices = [("a.weight", self.model.a.weight), ("b.weight", self.model.b.weight)]
        self.device = torch.device("cpu")

    def _grams(self, batches, micro, data):
        return sb.accumulate_grams(
            model=self.model, matrices=self.matrices, loader=data,
            batches=batches, kind="task", device=self.device,
            objective=_objective, micro_batches=micro,
        )

    def test_four_micro_batches_of_two_equal_one_batch_of_eight(self) -> None:
        wide = _batches(3, 8, seed=1)
        narrow = []
        for b in wide:
            for i in range(0, 8, 2):
                narrow.append({k: v[i:i + 2] for k, v in b.items()})

        wide_in, wide_out, _ = self._grams(3, 1, wide)
        thin_in, thin_out, _ = self._grams(3, 4, narrow)
        for name, _ in self.matrices:
            torch.testing.assert_close(thin_in[name], wide_in[name], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(thin_out[name], wide_out[name], rtol=1e-5, atol=1e-6)

    def test_default_is_one_micro_batch(self) -> None:
        data = _batches(4, 2, seed=2)
        explicit, _, _ = self._grams(4, 1, data)
        implicit, _, _ = sb.accumulate_grams(
            model=self.model, matrices=self.matrices, loader=data,
            batches=4, kind="task", device=self.device, objective=_objective,
        )
        for name, _ in self.matrices:
            torch.testing.assert_close(implicit[name], explicit[name])

    def test_one_probe_direction_per_aggregated_batch(self) -> None:
        # The rewind must hand every micro-batch the same r, as a single wide
        # batch does; without it the aggregate mixes several directions.
        seen: list[torch.Tensor] = []

        def probing(model, batch, device, generator=None):
            r = torch.randn(3, generator=generator)
            seen.append(r.clone())
            out = model(input_ids=batch["input_ids"])
            return (out.logits @ r).sum()

        generator = torch.Generator().manual_seed(7)
        sb.accumulate_grams(
            model=self.model, matrices=self.matrices, loader=_batches(8, 2, seed=3),
            batches=2, kind="reference", device=self.device,
            generator=generator, objective=probing, micro_batches=4,
        )
        self.assertEqual(len(seen), 8)
        for i in range(1, 4):
            torch.testing.assert_close(seen[i], seen[0])
        for i in range(5, 8):
            torch.testing.assert_close(seen[i], seen[4])
        self.assertFalse(torch.allclose(seen[0], seen[4]), "r non e cambiato fra i due termini")


class HeadWarmUp(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = _Toy()
        # A separable problem, so a head that learns anything must lower the loss.
        generator = torch.Generator().manual_seed(5)
        self.data = []
        for _ in range(12):
            labels = torch.randint(0, 3, (4,), generator=generator)
            inputs = torch.zeros(4, 6)
            inputs[torch.arange(4), labels] = 4.0
            self.data.append({"input_ids": inputs, "attention_mask": torch.ones(4, 6), "labels": labels})

    def _warm(self, steps):
        from merge_and_rebase.finetune import subspace_text as st
        messages: list[str] = []
        st.warm_up_head(
            model=self.model, loader=self.data, steps=steps, lr=1e-2,
            device=torch.device("cpu"), log=messages.append,
        )
        return messages

    def test_the_loss_falls_and_is_reported(self) -> None:
        messages = self._warm(60)
        final = [m for m in messages if "->" in m]
        self.assertTrue(final, f"nessun riepilogo: {messages}")
        before, after = (float(x) for x in final[-1].split("loss ")[1].split(" over")[0].split(" -> "))
        self.assertLess(after, before)

    def test_only_the_head_moves(self) -> None:
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self._warm(40)
        moved = [
            n for n, p in self.model.named_parameters()
            if not torch.allclose(p.detach(), before[n])
        ]
        self.assertEqual(moved, ["classification_head.weight"])

    def test_zero_steps_is_a_no_op(self) -> None:
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.assertEqual(self._warm(0), [])
        for n, p in self.model.named_parameters():
            torch.testing.assert_close(p.detach(), before[n])


class CacheKeyCoversTheNewKnobs(unittest.TestCase):
    BASE = dict(
        model_name="t5-base", pretrained="hf", task="rte", k=64, delta=0.1,
        one_sided=False, no_reference=False, batches=30,
        reference_datasets=("wikitext/wikitext-103-raw-v1",), seed=42,
    )

    def test_unset_knobs_leave_existing_keys_untouched(self) -> None:
        # The vision path passes neither, and its bases are already on disk.
        self.assertEqual(
            sb.cache_key(**self.BASE),
            sb.cache_key(**self.BASE, gram_batch_size=0, head_warmup_steps=0),
        )

    def test_each_knob_changes_the_key_and_shows_in_the_name(self) -> None:
        wide = sb.cache_key(**self.BASE, gram_batch_size=128)
        warm = sb.cache_key(**self.BASE, head_warmup_steps=300)
        both = sb.cache_key(**self.BASE, gram_batch_size=128, head_warmup_steps=300)
        self.assertEqual(len({sb.cache_key(**self.BASE), wide, warm, both}), 4)
        self.assertIn("_b128_", wide)
        self.assertIn("_w300_", warm)
        self.assertIn("_b128_w300_", both)


if __name__ == "__main__":
    unittest.main()
