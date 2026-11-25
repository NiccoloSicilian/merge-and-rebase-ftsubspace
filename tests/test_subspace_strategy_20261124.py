"""Checks for subspace-constrained expert training.

The eigenproblem, the reparameterisation and the strategy wiring are checked
without open_clip or a downloaded backbone: a small module with visual-tower
parameter names stands in for the vision model, which is all the strategy
actually requires.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from merge_and_rebase.finetune import subspace_bases as sb
from merge_and_rebase.finetune.strategies._subspace_parameterization import (
    bind_subspace_parameterization,
)
from merge_and_rebase.finetune.strategies.registry import get_strategy, list_strategies
from merge_and_rebase.finetune.strategies.subspace import _settings


class _TinyVisual(nn.Module):
    """Parameter names the visual-tower resolver recognises."""

    def __init__(self, width: int = 16, classes: int = 4) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([nn.Linear(width, width)])
        self.ln_pre = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.ln_pre(self.transformer.resblocks[0](images))
        return hidden @ self.proj


def _bases_for(named, names, rank=4):
    bases = {}
    for name, param in named:
        if name not in names:
            continue
        bases[name] = {
            "basis": torch.linalg.qr(torch.randn(param.shape[1], rank))[0],
            "output_basis": torch.linalg.qr(torch.randn(param.shape[0], rank))[0],
        }
    return bases


class EligibilityTest(unittest.TestCase):
    def test_only_two_dimensional_targets_are_selected(self):
        model = _TinyVisual()
        named = list(model.named_parameters())
        names = {name for name, _ in named}
        selected = {name for name, _ in sb.eligible_matrices(named, names)}
        self.assertEqual(selected, {"transformer.resblocks.0.weight", "proj"})

    def test_an_empty_selection_is_an_error(self):
        model = _TinyVisual()
        with self.assertRaisesRegex(RuntimeError, "zero matrices"):
            sb.eligible_matrices(list(model.named_parameters()), {"ln_pre.weight"})

    def test_rank_is_clamped_to_the_short_side(self):
        self.assertEqual(sb.rank_for((3072, 768), 64, False), 64)
        self.assertEqual(sb.rank_for((50, 768), 64, False), 50)
        self.assertEqual(sb.rank_for((50, 768), 64, True), 64)


class EigenproblemTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.size, self.rank = 32, 5
        samples = torch.randn(150, self.size)
        self.gram = samples.T @ samples / 150

    def test_identity_denominator_gives_the_ordinary_eigenvectors(self):
        basis, _ = sb.top_directions(self.gram, torch.eye(self.size), self.rank, 0.0)
        values, vectors = torch.linalg.eigh(self.gram.double())
        order = torch.argsort(values, descending=True)[: self.rank]
        self.assertAlmostEqual(
            sb.subspace_overlap(basis, vectors[:, order].float()), 1.0, places=4
        )

    def test_columns_are_orthonormal(self):
        basis, _ = sb.top_directions(self.gram, torch.eye(self.size), self.rank, 1e-2)
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(self.rank), atol=1e-5))

    def test_the_ridge_is_scale_free(self):
        plain, _ = sb.top_directions(self.gram, torch.eye(self.size), self.rank, 1e-2)
        scaled, _ = sb.top_directions(
            self.gram * 1000, torch.eye(self.size) * 1000, self.rank, 1e-2
        )
        self.assertGreater(sb.subspace_overlap(plain, scaled), 0.999)

    def test_a_singular_denominator_falls_back_instead_of_raising(self):
        denominator = torch.zeros(self.size, self.size)
        denominator[0, 0] = 1.0
        basis, _ = sb.top_directions(self.gram, denominator, self.rank, 0.0)
        self.assertEqual(tuple(basis.shape), (self.size, self.rank))
        self.assertTrue(torch.isfinite(basis).all())

    def test_the_generic_direction_is_avoided(self):
        denominator = torch.zeros(self.size, self.size)
        denominator[0, 0] = 100.0
        numerator = self.gram.clone()
        numerator[0, 0] += 100.0
        selective, _ = sb.top_directions(numerator, denominator, self.rank, 1e-2)
        plain, _ = sb.top_directions(numerator, torch.eye(self.size), self.rank, 1e-2)
        self.assertLess(selective[0].pow(2).sum().item(), plain[0].pow(2).sum().item())
        self.assertGreater(
            sb.energy_ratio(selective, numerator, denominator)[2],
            sb.energy_ratio(plain, numerator, denominator)[2],
        )

    def test_overlap_of_independent_subspaces_sits_near_k_over_n(self):
        size, rank = 256, 16
        left = torch.linalg.qr(torch.randn(size, rank))[0]
        right = torch.linalg.qr(torch.randn(size, rank))[0]
        self.assertAlmostEqual(sb.subspace_overlap(left, right), rank / size, delta=0.03)
        self.assertAlmostEqual(sb.subspace_overlap(left, left), 1.0, places=4)


class ParameterisationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _TinyVisual()
        self.named = list(self.model.named_parameters())
        self.targets = {"transformer.resblocks.0.weight", "proj"}
        self.before = {
            name: param.detach().clone() for name, param in self.named if name in self.targets
        }
        self.bases = _bases_for(self.named, self.targets)
        self.cores = bind_subspace_parameterization(
            model=self.model, named_params=self.named, bases=self.bases, device=torch.device("cpu")
        )

    def test_training_starts_at_the_pretrained_weights(self):
        current = self.model._current_param_map()
        for name, tensor in self.before.items():
            self.assertTrue(torch.allclose(current[name], tensor, atol=1e-6))
        self.assertEqual(self.model._subspace_module.core_norm(), 0.0)

    def test_the_cores_are_the_only_new_parameters(self):
        self.assertEqual(len(self.cores), 2)
        self.assertTrue(all(core.requires_grad for core in self.cores))
        for name, param in self.named:
            if name in self.targets:
                self.assertFalse(param.requires_grad)

    def test_gradients_reach_the_cores(self):
        self.model(torch.randn(3, 16)).pow(2).sum().backward()
        for core in self.cores:
            self.assertIsNotNone(core.grad)
            self.assertGreater(float(core.grad.abs().sum()), 0.0)

    def test_the_delta_stays_inside_the_subspace(self):
        with torch.no_grad():
            for core in self.cores:
                core.add_(torch.randn_like(core) * 0.05)
        current = self.model._current_param_map()
        for name, tensor in self.before.items():
            delta = current[name].detach() - tensor
            entry = self.bases[name]
            projected = entry["output_basis"].T @ delta @ entry["basis"]
            total = float(delta.pow(2).sum())
            self.assertAlmostEqual(float(projected.pow(2).sum()) / total, 1.0, places=4)
            self.assertLessEqual(int(torch.linalg.matrix_rank(delta, tol=1e-5)), 4)

    def test_core_norm_equals_the_delta_norm(self):
        with torch.no_grad():
            for core in self.cores:
                core.add_(torch.randn_like(core) * 0.05)
        current = self.model._current_param_map()
        measured = math.sqrt(
            sum(
                float((current[name].detach() - tensor).pow(2).sum())
                for name, tensor in self.before.items()
            )
        )
        self.assertAlmostEqual(measured, self.model._subspace_module.core_norm(), places=4)

    def test_the_materialised_state_dict_is_plain_and_carries_the_delta(self):
        with torch.no_grad():
            for core in self.cores:
                core.add_(torch.randn_like(core) * 0.05)
        expected = self.model._subspace_module.core_norm()
        state = self.model._materialized_state_dict()
        self.assertFalse(any(key.startswith("_subspace_module.") for key in state))
        for name in self.before:
            self.assertIn(name, state)
        carried = math.sqrt(
            sum(
                float((state[name] - tensor).pow(2).sum())
                for name, tensor in self.before.items()
            )
        )
        self.assertAlmostEqual(carried, expected, places=4)

    def test_a_mismatched_basis_is_refused(self):
        model = _TinyVisual()
        named = list(model.named_parameters())
        bad = {"proj": {"basis": torch.linalg.qr(torch.randn(7, 3))[0], "output_basis": None}}
        with self.assertRaisesRegex(RuntimeError, "rows but the weight has"):
            bind_subspace_parameterization(
                model=model, named_params=named, bases=bad, device=torch.device("cpu")
            )


class SettingsTest(unittest.TestCase):
    def test_a_reference_set_or_the_ablation_is_required(self):
        with self.assertRaisesRegex(ValueError, "reference_datasets"):
            _settings({"params": {"subspace": {"k": 64}}})
        _settings({"params": {"subspace": {"no_reference": True}}})
        _settings({"params": {"subspace": {"reference_datasets": ["ImageNet21KP"]}}})

    def test_misspelled_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _settings({"params": {"subspace": {"no_reference": True, "rnak": 8}}})

    def test_out_of_range_values_are_rejected(self):
        for bad in ({"k": 0}, {"delta": -1.0}, {"batches": 0}):
            with self.assertRaises(ValueError):
                _settings({"params": {"subspace": {**bad, "no_reference": True}}})

    def test_a_comma_separated_reference_list_is_accepted(self):
        resolved = _settings({"params": {"subspace": {"reference_datasets": "A, B"}}})
        self.assertEqual(resolved["reference_datasets"], ("A", "B"))

    def test_defaults(self):
        resolved = _settings({"params": {"subspace": {"no_reference": True}}})
        self.assertEqual(resolved["k"], 64)
        self.assertEqual(resolved["delta"], 1e-2)
        self.assertEqual(resolved["batches"], 30)
        self.assertFalse(resolved["one_sided"])
        self.assertFalse(resolved["include_non_matrices"])


class StrategyWiringTest(unittest.TestCase):
    """configure() end to end on the ablation path, which needs no reference set."""

    class _Loaders:
        def __init__(self, train):
            self.train = train

    def _loader(self, width=16, classes=4, batches=4):
        return [
            (torch.randn(5, width), torch.randint(0, classes, (5,))) for _ in range(batches)
        ]

    def test_the_strategy_is_registered(self):
        self.assertIn("subspace", list_strategies())

    def test_configure_returns_an_optimizer_over_the_cores(self):
        torch.manual_seed(0)
        model = _TinyVisual()
        with TemporaryDirectory() as directory:
            opt, scheduler, info = get_strategy("subspace").configure(
                model=model,
                lr=1e-4,
                weight_decay=0.0,
                warmup_length=0,
                scheduler_name="cosine",
                optimizer="adamw",
                steps=10,
                device=torch.device("cpu"),
                strategy_cfg={
                    "params": {
                        "trainable_params": "all_trainable",
                        "subspace": {"k": 4, "no_reference": True, "batches": 2},
                    }
                },
                task="Tiny",
                loaders=self._loaders_for(self._loader()),
                build_cfg=None,
                batch_size=5,
                num_workers=0,
                val_fraction=0.1,
                seed=42,
                out_dir=Path(directory),
            )
        self.assertEqual(info["parameterization"], "subspace")
        self.assertEqual(info["subspace_matrices"], 2)
        self.assertEqual(info["subspace_k"], 4)
        self.assertTrue(info["subspace_no_reference"])
        # 2 matrices x a 4x4 core
        self.assertEqual(info["delta_params"], 32)
        self.assertEqual(info["trainable_params"], 32)
        self.assertEqual(len(opt.param_groups[0]["params"]), 2)
        self.assertTrue(callable(scheduler))
        self.assertTrue(hasattr(model, "_materialized_state_dict"))

    def test_training_moves_the_weights_only_inside_the_subspace(self):
        torch.manual_seed(0)
        model = _TinyVisual()
        before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in {"transformer.resblocks.0.weight", "proj"}
        }
        with TemporaryDirectory() as directory:
            opt, _scheduler, _info = get_strategy("subspace").configure(
                model=model,
                lr=1e-2,
                weight_decay=0.0,
                warmup_length=0,
                scheduler_name="cosine",
                optimizer="adamw",
                steps=5,
                device=torch.device("cpu"),
                strategy_cfg={
                    "params": {
                        "trainable_params": "all_trainable",
                        "subspace": {"k": 4, "no_reference": True, "batches": 2},
                    }
                },
                task="Tiny",
                loaders=self._loaders_for(self._loader()),
                build_cfg=None,
                batch_size=5,
                num_workers=0,
                val_fraction=0.1,
                seed=42,
                out_dir=Path(directory),
            )
            for images, labels in self._loader(batches=3):
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(images), labels).backward()
                opt.step()
        state = model._materialized_state_dict()
        moved = False
        for name, tensor in before.items():
            delta = state[name] - tensor
            if float(delta.abs().max()) > 0:
                moved = True
                self.assertLessEqual(int(torch.linalg.matrix_rank(delta, tol=1e-6)), 4)
        self.assertTrue(moved, "training did not move any covered matrix")

    def test_a_missing_loader_is_reported(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "needs the task train loader"):
                get_strategy("subspace").configure(
                    model=_TinyVisual(),
                    lr=1e-4,
                    weight_decay=0.0,
                    warmup_length=0,
                    steps=10,
                    device=torch.device("cpu"),
                    strategy_cfg={
                        "params": {"subspace": {"k": 4, "no_reference": True, "batches": 2}}
                    },
                    task="Tiny",
                    loaders=None,
                    out_dir=Path(directory),
                )

    def test_bases_are_cached_and_reused(self):
        torch.manual_seed(0)
        with TemporaryDirectory() as directory:
            kwargs = dict(
                lr=1e-4,
                weight_decay=0.0,
                warmup_length=0,
                steps=10,
                device=torch.device("cpu"),
                strategy_cfg={
                    "params": {
                        "trainable_params": "all_trainable",
                        "subspace": {"k": 4, "no_reference": True, "batches": 2},
                    }
                },
                task="Tiny",
                loaders=self._loaders_for(self._loader()),
                build_cfg=None,
                batch_size=5,
                num_workers=0,
                val_fraction=0.1,
                seed=42,
                out_dir=Path(directory),
            )
            get_strategy("subspace").configure(model=_TinyVisual(), **kwargs)
            cached = sorted(Path(directory).glob("subspace_bases/*.pt"))
            self.assertEqual(len(cached), 1)
            stamp = cached[0].stat().st_mtime_ns
            get_strategy("subspace").configure(model=_TinyVisual(), **kwargs)
            self.assertEqual(cached[0].stat().st_mtime_ns, stamp)

    def _loaders_for(self, train):
        return StrategyWiringTest._Loaders(train)


if __name__ == "__main__":
    unittest.main()


class ReferenceDatasetTest(unittest.TestCase):
    """The reference set must be resolvable and must not be a merged task."""

    def test_the_imagenet_mirror_resolves(self):
        from merge_and_rebase.eval.datasets.vision8_14_20 import (
            VISION8_TASKS,
            VISION_SUPPORTED_TASKS,
            _vision_spec,
        )

        self.assertIn("ImageNet1KWds", VISION_SUPPORTED_TASKS)
        path, config, splits = _vision_spec("ImageNet1KWds")
        self.assertEqual(path, "clip-benchmark/wds_imagenet1k")
        self.assertIsNone(config)
        self.assertEqual(set(splits), {"train", "test"})
        # Both keys point at the mirror's small split on purpose: this task only
        # ever supplies images for statistics.
        self.assertEqual(set(splits.values()), {"test"})
        # A reference set that is one of the merged tasks would leak.
        self.assertNotIn("ImageNet1KWds", VISION8_TASKS)

    def test_the_shipped_config_uses_a_resolvable_reference(self):
        import re

        from merge_and_rebase.eval.datasets.vision8_14_20 import (
            VISION8_TASKS,
            _vision_spec,
        )

        config = (
            Path(__file__).resolve().parents[1]
            / "src/merge_and_rebase/finetune/configs/SUBSPACE"
            / "vision-subspace-vitb32-vision8.yaml"
        )
        text = config.read_text()
        block = re.search(r"reference_datasets:\s*\n((?:\s*-\s*\S+\n)+)", text)
        self.assertIsNotNone(block, "the preset declares no reference_datasets")
        names = re.findall(r"-\s*(\S+)", block.group(1))
        self.assertTrue(names)
        for name in names:
            _vision_spec(name)
            self.assertNotIn(name, VISION8_TASKS)


class QuickGeluGuardTest(unittest.TestCase):
    """The OpenAI weights need QuickGELU, and open_clip only warns about it."""

    def _model(self, activation: nn.Module):
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.gelu = activation
        transformer = nn.Module()
        transformer.resblocks = nn.ModuleList([block])
        visual = nn.Module()
        visual.transformer = transformer
        model = nn.Module()
        model.visual = visual
        return model

    def test_plain_gelu_is_refused(self):
        from merge_and_rebase.models.openclip_classifier import _require_quick_gelu

        with self.assertRaisesRegex(RuntimeError, "instead of QuickGELU"):
            _require_quick_gelu(self._model(nn.GELU()), "ViT-B-32", "openai")

    def test_quick_gelu_is_accepted(self):
        from merge_and_rebase.models.openclip_classifier import _require_quick_gelu

        class QuickGELU(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(1.702 * x)

        _require_quick_gelu(self._model(QuickGELU()), "ViT-B-32", "openai")

    def test_an_unrecognised_shape_is_not_a_failure(self):
        from merge_and_rebase.models.openclip_classifier import _require_quick_gelu

        _require_quick_gelu(nn.Module(), "custom", "openai")

    def test_the_loader_forces_quick_gelu_for_the_openai_tag(self):
        import inspect
        import re

        from merge_and_rebase.models import openclip_classifier

        source = inspect.getsource(openclip_classifier)
        # A bare `quick_gelu=` keyword lands in **model_kwargs and does nothing,
        # so match it only when it is not the tail of `force_quick_gelu=`.
        self.assertIsNone(re.search(r"(?<!force_)quick_gelu=quick_gelu", source))
        self.assertIn("force_quick_gelu=quick_gelu,", source)
        self.assertIn("force_quick_gelu=True,", source)


class ReferenceLoaderColumnTest(unittest.TestCase):
    """The reference set must work without a label space."""

    def test_webdataset_columns_are_detected(self):
        from merge_and_rebase.finetune.strategies.subspace import (
            _IMAGE_COLUMNS,
            _LABEL_COLUMNS,
        )

        # The mirrors this repo already uses expose jpg/cls, not image/label.
        self.assertIn("jpg", _IMAGE_COLUMNS)
        self.assertIn("cls", _LABEL_COLUMNS)
        # The plain HF convention still wins when both are present.
        self.assertLess(_IMAGE_COLUMNS.index("image"), _IMAGE_COLUMNS.index("jpg"))
        self.assertLess(_LABEL_COLUMNS.index("label"), _LABEL_COLUMNS.index("cls"))

    def test_class_names_are_never_required(self):
        import inspect

        from merge_and_rebase.finetune.strategies import subspace

        source = inspect.getsource(subspace._reference_loader)
        # A placeholder override skips extraction; strict would raise on a
        # mirror whose integer labels carry no ClassLabel feature.
        self.assertIn('classnames_override=("generic",)', source)
        self.assertIn("strict_classnames=False", source)

    def test_it_does_not_use_the_regularizer_helper(self):
        import inspect

        from merge_and_rebase.finetune.strategies import subspace

        source = inspect.getsource(subspace)
        self.assertNotIn("build_vision_regularizer_task_context", source)
