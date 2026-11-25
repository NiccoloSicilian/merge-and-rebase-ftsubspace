"""Rebuilding heads.pt from per-task checkpoints.

train_text writes heads.pt with a full overwrite, so training one task per job
leaves a file holding a single task. These checks cover the recovery path,
including its refusal to produce a partial head set.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from merge_and_rebase.finetune.collect_heads import collect, extract_head


def _state(labels: int = 2, width: int = 8, prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(width, width),
        f"{prefix}shared.weight": torch.randn(16, width),
        f"{prefix}classification_head.dense.weight": torch.randn(width, width),
        f"{prefix}classification_head.dense.bias": torch.randn(width),
        f"{prefix}classification_head.out_proj.weight": torch.randn(labels, width),
        f"{prefix}classification_head.out_proj.bias": torch.randn(labels),
    }


class ExtractTest(unittest.TestCase):
    def test_only_the_head_is_taken(self):
        head = extract_head(_state())
        self.assertEqual(
            set(head),
            {
                "classification_head.dense.weight",
                "classification_head.dense.bias",
                "classification_head.out_proj.weight",
                "classification_head.out_proj.bias",
            },
        )

    def test_the_peft_prefix_is_canonicalised(self):
        # base_model.model. is what PEFT adds; the merge expects it gone.
        head = extract_head(_state(prefix="base_model.model."))
        self.assertIn("classification_head.out_proj.weight", head)

    def test_a_score_root_is_recognised(self):
        state = {"score.weight": torch.randn(2, 8), "encoder.x.weight": torch.randn(8, 8)}
        self.assertEqual(set(extract_head(state)), {"score.weight"})

    def test_a_checkpoint_without_a_head_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "no classification head"):
            extract_head({"encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(4, 4)})

    def test_the_tensors_are_detached_copies(self):
        state = _state()
        state["classification_head.dense.bias"].requires_grad_(True)
        head = extract_head(state)
        self.assertFalse(head["classification_head.dense.bias"].requires_grad)
        head["classification_head.dense.bias"].zero_()
        self.assertGreater(float(state["classification_head.dense.bias"].abs().sum()), 0.0)


class CollectTest(unittest.TestCase):
    def _tree(self, directory: Path, tasks: dict[str, int], name: str):
        for task, labels in tasks.items():
            task_dir = directory / task
            task_dir.mkdir(parents=True)
            torch.save({"state_dict": _state(labels)}, task_dir / name)

    def test_one_head_per_task(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root, {"rte": 2, "snli": 3, "scitail": 2}, "subspace_best_ep.pt")
            heads = collect(root, "subspace_best_ep.pt")
            self.assertEqual(set(heads), {"rte", "snli", "scitail"})
            # The label space differs per task, which is why heads are not merged.
            self.assertEqual(tuple(heads["snli"]["classification_head.out_proj.weight"].shape)[0], 3)
            self.assertEqual(tuple(heads["rte"]["classification_head.out_proj.weight"].shape)[0], 2)

    def test_a_requested_task_without_a_checkpoint_is_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root, {"rte": 2}, "subspace_best_ep.pt")
            # A merge over five of six heads is a wrong result, not a partial one.
            with self.assertRaisesRegex(RuntimeError, "no subspace_best_ep.pt for"):
                collect(root, "subspace_best_ep.pt", ["rte", "snli"])

    def test_tasks_without_a_checkpoint_are_skipped_when_not_requested(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root, {"rte": 2}, "subspace_best_ep.pt")
            (root / "half_done").mkdir()
            heads = collect(root, "subspace_best_ep.pt")
            self.assertEqual(set(heads), {"rte"})

    def test_an_empty_tree_is_reported(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "no head found under"):
                collect(Path(tmp), "subspace_best_ep.pt")

    def test_a_bare_state_dict_payload_works(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rte").mkdir(parents=True)
            torch.save(_state(2), root / "rte" / "full_best_ep.pt")
            heads = collect(root, "full_best_ep.pt")
            self.assertIn("classification_head.dense.weight", heads["rte"])


if __name__ == "__main__":
    unittest.main()


class CanonicalisationParityTest(unittest.TestCase):
    """The copied canonicaliser must match the original."""

    @staticmethod
    def _original_available() -> bool:
        try:
            from merge_and_rebase.finetune.train_text import (  # noqa: F401
                _canonical_param_name,
            )
        except Exception:
            return False
        return True

    def test_the_copy_matches_the_original(self):
        if not self._original_available():
            self.skipTest("train_text needs the data stack")
        from merge_and_rebase.finetune.collect_heads import canonical_param_name
        from merge_and_rebase.finetune.train_text import _canonical_param_name

        for name in (
            "classification_head.out_proj.weight",
            "base_model.model.classification_head.dense.bias",
            "base_model.model.score.modules_to_save.default.weight",
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            "score",
        ):
            self.assertEqual(canonical_param_name(name), _canonical_param_name(name), name)

    def test_the_copy_behaves_as_documented(self):
        from merge_and_rebase.finetune.collect_heads import canonical_param_name

        self.assertEqual(canonical_param_name("base_model.model.score.weight"), "score.weight")
        self.assertEqual(
            canonical_param_name("score.modules_to_save.default.weight"), "score.weight"
        )
        self.assertEqual(canonical_param_name("plain.name"), "plain.name")


class SidecarTest(unittest.TestCase):
    """Per-task sidecars are what makes one-task-per-job safe."""

    def test_sidecars_are_preferred_over_checkpoints(self):
        from merge_and_rebase.finetune.collect_heads import collect

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rte").mkdir()
            torch.save({"state_dict": _state(2)}, root / "rte" / "subspace_best_ep.pt")
            (root / "heads").mkdir()
            marker = {"classification_head.out_proj.weight": torch.full((2, 8), 7.0)}
            torch.save(marker, root / "heads" / "rte.pt")
            heads = collect(root, "subspace_best_ep.pt")
            self.assertEqual(
                float(heads["rte"]["classification_head.out_proj.weight"][0, 0]), 7.0
            )

    def test_a_sidecar_without_a_task_directory_still_counts(self):
        from merge_and_rebase.finetune.collect_heads import collect

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "heads").mkdir(parents=True)
            torch.save({"score.weight": torch.randn(2, 8)}, root / "heads" / "snli.pt")
            self.assertEqual(set(collect(root, "x.pt")), {"snli"})

    def test_trainer_writes_one_file_per_task(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/train_text.py"
        ).read_text()
        # Without the sidecars, parallel jobs overwrite one another's heads.
        self.assertIn('sidecar_dir = heads_path.parent / "heads"', source)
        self.assertIn('torch.save(head, sidecar_dir / f"{task_name}.pt")', source)
        # And the combined file folds in every sidecar, not just this run's.
        self.assertIn('for sidecar in sorted(sidecar_dir.glob("*.pt")):', source)
