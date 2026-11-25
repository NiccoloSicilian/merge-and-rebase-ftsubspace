"""Merge-aware training by confining each expert update to a selected subspace.

``strategy.name = "subspace"``. Every eligible matrix is reparametrised as
``W = W_0 + U C V^T`` with ``U`` and ``V`` frozen bases fixed before training
and ``C`` the only trainable tensor, so the delta is rank <= k with its row and
column spaces pinned by construction rather than by penalty.

The bases come from gradient statistics at the pre-trained weights: the Grams
of the task's own gradients against the Grams of a reference set that is not
one of the tasks being merged. See ``finetune/subspace_bases.py`` for the
eigenproblem and its diagnostics.

Configuration, under ``strategy.params.subspace``:

    k                   subspace dimension per side (default 64)
    delta               ridge on the reference Gram (default 0.01)
    batches             gradient batches for the task statistics (default 30)
    reference_batches   gradient batches for the reference set (default: batches)
    reference_datasets  reference task names, e.g. ["ImageNet21KP"]
    one_sided           restrict only the input side, delta = C V^T
    no_reference        ablation: use the identity instead of the reference Grams
    include_non_matrices  also train the non-2-D tensors in the trainable set
    cache_dir           where bases are cached (default: <out_dir>/subspace_bases)
    rebuild             recompute even when a cached file exists
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from merge_and_rebase.finetune.schedulers import build_lr_scheduler
from merge_and_rebase.models.parameter_subsets import (
    _visual_local_name,
    resolve_parameter_subset,
)

from .. import subspace_bases as sb
from ._subspace_parameterization import bind_subspace_parameterization
from .base import build_optimizer
from .registry import register


_DEFAULTS: dict[str, Any] = {
    "k": 64,
    "delta": 1e-2,
    "batches": 30,
    "reference_batches": 0,
    "reference_datasets": (),
    "one_sided": False,
    "no_reference": False,
    "include_non_matrices": False,
    "cache_dir": None,
    "rebuild": False,
}


@dataclass(frozen=True)
class SubspaceFinetune:
    name: str = "subspace"

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_length: int,
        scheduler_name: str = "cosine",
        optimizer: str = "adamw",
        steps: int,
        device: torch.device,
        strategy_cfg: dict | None = None,
        task: str | None = None,
        loaders: Any | None = None,
        build_cfg: Any | None = None,
        batch_size: int = 64,
        num_workers: int = 6,
        val_fraction: float = 0.1,
        seed: int = 42,
        out_dir: Path | str | None = None,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        del kwargs
        settings = _settings(strategy_cfg)
        prefix = f"[{task}] subspace" if task else "subspace"

        def log(message: str) -> None:
            print(f"{prefix}: {message}", flush=True)

        trainable_mode = str(
            _cfg_value(strategy_cfg, "trainable_params", "all_trainable")
        ).strip().lower()
        if trainable_mode not in {"all_trainable", "regularized_only"}:
            raise ValueError(
                "strategy.params.trainable_params must be one of: all_trainable, regularized_only"
            )

        named_params = list(model.named_parameters())
        effective = {name for name, param in named_params if param.requires_grad}
        mode_supported = trainable_mode == "all_trainable"
        if trainable_mode == "regularized_only":
            resolution = resolve_parameter_subset(model, "regularized_only")
            if resolution.supported:
                effective &= set(resolution.parameter_names)
                mode_supported = True

        # Restrict to the visual tower whatever the mode says. Under
        # all_trainable the text tower's token and position embeddings are 2-D
        # and would otherwise be swept in, which would put text parameters into
        # every task vector.
        visual = {name for name, _ in named_params if _visual_local_name(name) is not None}
        if not visual:
            raise RuntimeError(
                "the subspace strategy found no visual-tower parameters; it only "
                "supports vision backbones"
            )
        dropped = len(effective - visual)
        effective &= visual
        if dropped:
            log(f"ignoring {dropped} non-visual trainable tensors")

        matrices = sb.eligible_matrices(named_params, effective)
        log(
            f"{len(matrices)} eligible matrices out of {len(effective)} trainable "
            f"tensors, k={settings['k']}"
            + (" (one-sided)" if settings["one_sided"] else " (both sides)")
        )

        bases = _bases(
            model=model,
            matrices=matrices,
            settings=settings,
            task=task,
            loaders=loaders,
            build_cfg=build_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            val_fraction=val_fraction,
            seed=seed,
            out_dir=out_dir,
            device=device,
            log=log,
        )

        # Freeze everything the subspace covers, then everything else unless it
        # was explicitly asked for. The non-matrix tensors -- the convolution
        # stem, the LayerNorms, the biases -- are shared by every expert, so
        # leaving them at the pre-trained values is what lets the matrices alone
        # carry the task.
        extra: list[nn.Parameter] = []
        matrix_names = {name for name, _ in matrices}
        for name, param in named_params:
            if name in matrix_names:
                param.requires_grad_(False)
                continue
            keep = settings["include_non_matrices"] and name in effective
            param.requires_grad_(keep)
            if keep:
                extra.append(param)

        before = {name: param.detach().clone() for name, param in matrices}
        cores = bind_subspace_parameterization(
            model=model,
            named_params=named_params,
            bases={name: bases[name] for name, _ in matrices},
            device=device,
        )
        drift = _drift(model, before)
        log(
            f"{sum(core.numel() for core in cores)} trainable core parameters; "
            f"max |W - W_0| at init = {drift:.3e} {'[ok]' if drift < 1e-5 else '[FAIL]'}"
        )
        if drift >= 1e-5:
            raise RuntimeError(
                "the subspace parameterisation does not start at the pre-trained "
                "weights; refusing to train"
            )

        opt_params = list(cores) + extra
        if not opt_params:
            raise RuntimeError("subspace strategy produced zero trainable parameters.")
        opt = build_optimizer(opt_params, optimizer, lr, weight_decay)
        scheduler = build_lr_scheduler(
            opt, name=scheduler_name, base_lrs=lr, warmup_length=warmup_length, steps=steps
        )
        info: dict[str, Any] = {
            "trainable_params": int(sum(p.numel() for p in opt_params)),
            "delta_params": int(sum(core.numel() for core in cores)),
            "trainable_params_fallback": int(
                trainable_mode == "regularized_only" and not mode_supported
            ),
            "parameterization": "subspace",
            "trainable_params_mode": trainable_mode,
            "trainable_params_effective": "all_trainable" if not mode_supported else trainable_mode,
            "scheduler_name": scheduler_name,
            "subspace_matrices": len(matrices),
            "subspace_k": int(settings["k"]),
            "subspace_delta": float(settings["delta"]),
            "subspace_one_sided": bool(settings["one_sided"]),
            "subspace_no_reference": bool(settings["no_reference"]),
            "subspace_non_matrices": len(extra),
        }
        return opt, scheduler, info


def _settings(strategy_cfg: dict | None) -> dict[str, Any]:
    raw = _cfg_value(strategy_cfg, "subspace", {})
    if not isinstance(raw, dict):
        raise ValueError("strategy.params.subspace must be a mapping")
    unknown = set(raw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unsupported strategy.params.subspace options: {sorted(unknown)}. "
            f"Available: {sorted(_DEFAULTS)}"
        )
    settings = dict(_DEFAULTS)
    settings.update(raw)
    settings["k"] = int(settings["k"])
    if settings["k"] < 1:
        raise ValueError("strategy.params.subspace.k must be positive")
    settings["delta"] = float(settings["delta"])
    if settings["delta"] < 0:
        raise ValueError("strategy.params.subspace.delta must not be negative")
    settings["batches"] = int(settings["batches"])
    settings["reference_batches"] = int(settings["reference_batches"])
    if settings["batches"] < 1:
        raise ValueError("strategy.params.subspace.batches must be positive")
    references = settings["reference_datasets"]
    if isinstance(references, str):
        references = [item.strip() for item in references.split(",") if item.strip()]
    settings["reference_datasets"] = tuple(references or ())
    settings["one_sided"] = bool(settings["one_sided"])
    settings["no_reference"] = bool(settings["no_reference"])
    settings["include_non_matrices"] = bool(settings["include_non_matrices"])
    settings["rebuild"] = bool(settings["rebuild"])
    if not settings["no_reference"] and not settings["reference_datasets"]:
        raise ValueError(
            "strategy.params.subspace needs reference_datasets (a task NOT being "
            "merged, e.g. [\"ImageNet21KP\"]), or no_reference=true to run the "
            "ablation without a reference set"
        )
    return settings


def _cfg_value(strategy_cfg: dict | None, key: str, default: Any) -> Any:
    if not isinstance(strategy_cfg, dict):
        return default
    params = strategy_cfg.get("params", None)
    if isinstance(params, dict) and key in params:
        return params[key]
    return strategy_cfg.get(key, default)


def _drift(model: nn.Module, before: dict[str, torch.Tensor]) -> float:
    """max |W - W_0| over the covered matrices, read through the bound forward."""

    getter = getattr(model, "_current_param_map", None)
    if not callable(getter):
        return 0.0
    current = getter()
    with torch.no_grad():
        return max(
            float((current[name].detach().to(tensor.device) - tensor).abs().max())
            for name, tensor in before.items()
        )


_IMAGE_COLUMNS = ("image", "jpg", "png", "webp", "img", "jpeg")
_LABEL_COLUMNS = ("label", "cls", "class", "fine_label", "coarse_label")


def _reference_loader(
    *,
    reference: str,
    model: nn.Module,
    build_cfg: Any,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    log: Callable[[str], None],
):
    """A loader of reference images, built without demanding a label space.

    The regularizers' own helper goes through ``build_vision_loaders`` with its
    defaults, which require an ``image``/``label`` column pair and a ClassLabel
    feature to read class names from. Webdataset mirrors use ``jpg``/``cls``
    with plain integer labels and fail that check.

    The reference statistics are label-free by construction -- they
    differentiate ``r^T f(x)`` for a random ``r`` -- so neither the label column
    nor the class names are ever read. The columns are therefore detected from
    the dataset features, and a single placeholder class name is passed so the
    extraction is skipped entirely rather than scanning rows to guess a class
    count that nothing will use.
    """

    from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
    from merge_and_rebase.eval.datasets.vision8_14_20 import _vision_spec

    hf_path, hf_config, split_map = _vision_spec(reference)
    hf_ds = load_hf_splits(
        hf_path, config=hf_config,
        requested_splits=tuple(dict.fromkeys(split_map.values())),
    )
    columns = set()
    for split in hf_ds.values():
        columns |= set(split.features)
    image_key = next((name for name in _IMAGE_COLUMNS if name in columns), None)
    label_key = next((name for name in _LABEL_COLUMNS if name in columns), None)
    if image_key is None:
        raise RuntimeError(
            f"{reference} ({hf_path}) exposes no recognisable image column; "
            f"found {sorted(columns)}"
        )
    if label_key is None:
        # build_vision_loaders needs a column name to hand to its collate; any
        # present column works because the value is never read.
        label_key = next(iter(sorted(columns - {image_key})), image_key)
    log(f"reference {reference}: columns image={image_key!r} label={label_key!r}")

    # Reuse the preprocessing of the model already on the GPU when it exposes
    # it, rather than loading a second copy of CLIP just for a transform.
    owner = getattr(model, "module", model)
    classifier = getattr(owner, "clip_model", None)
    preprocess = getattr(classifier, "preprocess", None)
    if preprocess is None:
        from merge_and_rebase.models.openclip_classifier import OpenClipClassifier

        preprocess = OpenClipClassifier.build(build_cfg).preprocess

    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        val_fraction=val_fraction,
        seed=seed,
        image_key=image_key,
        label_key=label_key,
        classnames_override=("generic",),
        strict_classnames=False,
    )
    return loaders.train


def _bases(
    *,
    model: nn.Module,
    matrices: list[tuple[str, nn.Parameter]],
    settings: dict[str, Any],
    task: str | None,
    loaders: Any | None,
    build_cfg: Any | None,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    out_dir: Path | str | None,
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict[str, torch.Tensor | None]]:
    """Load the cached bases, or compute them from gradient statistics."""

    model_name = getattr(build_cfg, "model_name", "model")
    pretrained = getattr(build_cfg, "pretrained", "pretrained")
    cache_root = Path(
        settings["cache_dir"]
        if settings["cache_dir"]
        else (Path(out_dir) if out_dir else Path("."))
    )
    if not settings["cache_dir"]:
        cache_root = cache_root / "subspace_bases"
    path = cache_root / sb.cache_key(
        model_name=str(model_name),
        pretrained=str(pretrained),
        task=str(task),
        k=settings["k"],
        delta=settings["delta"],
        one_sided=settings["one_sided"],
        no_reference=settings["no_reference"],
        batches=settings["batches"],
        reference_datasets=settings["reference_datasets"],
        seed=seed,
    )
    if not settings["rebuild"]:
        cached = sb.load_cached(path, matrices)
        if cached is not None:
            log(f"bases loaded from {path}")
            return cached

    if loaders is None or getattr(loaders, "train", None) is None:
        raise RuntimeError(
            "the subspace strategy needs the task train loader to compute its "
            "bases; train_vision must pass loaders= into Strategy.configure()"
        )

    # Gradients are needed on the eligible matrices and nowhere else.
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for _, param in matrices:
        param.requires_grad_(True)

    log(f"task gradients: {settings['batches']} batches")
    task_inputs, task_outputs, seen = sb.accumulate_grams(
        model=model,
        matrices=matrices,
        loader=loaders.train,
        batches=settings["batches"],
        kind="task",
        device=device,
        log=log,
    )
    dead = [name for name, _ in matrices if float(torch.diagonal(task_inputs[name]).sum()) <= 0]
    if dead:
        # The eigenvectors of a zero matrix are arbitrary, so the basis would be
        # meaningless and training would silently run in a random subspace.
        raise RuntimeError(
            f"{len(dead)} matrix/matrices received no gradient from the task loss: "
            f"{dead[:8]}. Check that they are on the forward path."
        )

    reference_batches = settings["reference_batches"] or settings["batches"]
    if settings["no_reference"]:
        log("no_reference: the denominator is the identity")
        reference_inputs = {
            name: torch.eye(param.shape[1], device=device) for name, param in matrices
        }
        reference_outputs = {
            name: torch.eye(param.shape[0], device=device) for name, param in matrices
        }
    else:
        reference_inputs = {
            name: torch.zeros(param.shape[1], param.shape[1], device=device)
            for name, param in matrices
        }
        reference_outputs = {
            name: torch.zeros(param.shape[0], param.shape[0], device=device)
            for name, param in matrices
        }
        if build_cfg is None:
            raise RuntimeError(
                "the subspace strategy needs build_cfg to construct its reference "
                "loader; train_vision must pass build_cfg= into Strategy.configure()"
            )
        for index, reference in enumerate(settings["reference_datasets"]):
            log(f"reference gradients on {reference}: {reference_batches} batches")
            reference_loader = _reference_loader(
                reference=reference,
                model=model,
                build_cfg=build_cfg,
                batch_size=batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
                seed=seed,
                log=log,
            )
            generator = torch.Generator(device=device).manual_seed(seed + 1 + index)
            # The loader comes from the reference context, but the gradients are
            # taken through THIS model: a fresh classifier would give the right
            # images and the wrong parameters.
            partial_inputs, partial_outputs, _ = sb.accumulate_grams(
                model=model,
                matrices=matrices,
                loader=reference_loader,
                batches=reference_batches,
                kind="reference",
                device=device,
                generator=generator,
                log=log,
            )
            for name, _ in matrices:
                reference_inputs[name] += partial_inputs[name]
                reference_outputs[name] += partial_outputs[name]
        count = max(len(settings["reference_datasets"]), 1)
        for name, _ in matrices:
            reference_inputs[name] /= count
            reference_outputs[name] /= count

    log("solving the eigenproblems (fp64, on CPU)")
    bases, report = sb.solve_bases(
        matrices=matrices,
        task_inputs=task_inputs,
        task_outputs=task_outputs,
        reference_inputs=reference_inputs,
        reference_outputs=reference_outputs,
        k=settings["k"],
        delta=settings["delta"],
        one_sided=settings["one_sided"],
        seed=seed,
        no_reference=settings["no_reference"],
        log=log,
    )
    sb.save_cached(
        path,
        bases=bases,
        report=report,
        meta={
            "task": task,
            "model_name": str(model_name),
            "pretrained": str(pretrained),
            "k": settings["k"],
            "delta": settings["delta"],
            "one_sided": settings["one_sided"],
            "no_reference": settings["no_reference"],
            "task_batches": seen,
            "reference_batches": 0 if settings["no_reference"] else reference_batches,
            "reference_datasets": list(settings["reference_datasets"]),
            "batch_size": batch_size,
            "seed": seed,
            "matrices": [name for name, _ in matrices],
        },
    )
    log(f"bases saved to {path}")
    del task_inputs, task_outputs, reference_inputs, reference_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bases


register(SubspaceFinetune())
