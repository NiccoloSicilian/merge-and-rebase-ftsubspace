"""Selected subspaces for merge-aware expert training.

Each eligible matrix is confined to a subspace fixed before training:

    W = W_0 + U C V^T

with ``U`` and ``V`` frozen and ``C`` the only trainable tensor. The delta is
therefore rank <= k with its row and column spaces pinned, by construction
rather than by penalty. Where DELTA distils linearised behaviour and TAK
penalises curvature-weighted drift, this restricts the hypothesis space.

Choosing the bases
------------------
For each matrix, two gradient Grams are accumulated on the task's own data,

    A_j = E[G^T G]   (n x n, input side)     S_j = E[G G^T]   (m x m, output side)

and the same two on a reference set that is not one of the tasks being merged,
giving ``A_g`` and ``S_g``. The bases are the top-k solutions of the generalised
symmetric-definite eigenproblem

    A_j v = lambda C v,     C = A_g + delta * (tr A_g / n) * I

i.e. the directions the task wants to move along *relative to* the directions
the pre-trained model already uses generically. Those are the ones the other
experts are least likely to claim, which is what makes the resulting task
vectors composable. The same is solved with ``(S_j, S_g)`` for the output side.

The ridge is scaled by the mean eigenvalue of the denominator, so it is
scale-free. It is not cosmetic: where the generic energy is near zero the ratio
diverges and the top eigenvectors would be estimation noise.

``no_reference`` replaces the generic Grams with the identity, which collapses
the generalised problem into the ordinary one: the bases become the top-k
eigenvectors of the task Grams alone, with no selectivity. The ablation
therefore runs through this same code rather than a parallel path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── eligibility ────────────────────────────────────────────────────────────


def eligible_matrices(
    named_params: Sequence[tuple[str, nn.Parameter]],
    target_names: Iterable[str] | None = None,
) -> list[tuple[str, nn.Parameter]]:
    """The 2-D parameters among ``target_names``, in their declared order.

    Restricting to matrices is what makes this comparable with the SVD-based
    mergers, which act on exactly that set. On an open_clip ViT-B/32 visual
    tower it selects 50 tensors: twelve blocks of
    {in_proj, out_proj, c_fc, c_proj} plus the position embedding and the
    output projection. ``conv1`` is 4-D and the LayerNorms and biases are 1-D,
    so none of them are here.
    """

    allowed = None if target_names is None else set(target_names)
    selected = [
        (name, param)
        for name, param in named_params
        if param.dim() == 2 and (allowed is None or name in allowed)
    ]
    if not selected:
        raise RuntimeError(
            "the subspace strategy selected zero matrices; check "
            "strategy.params.trainable_params and that the target module has "
            "2-D weights"
        )
    return selected


def rank_for(shape: tuple[int, int], k: int, one_sided: bool) -> int:
    """``k`` clamped to the tensor, so a short side cannot be asked for more
    directions than it has. A 50x768 position embedding lands on 50."""

    rows, columns = shape
    return min(k, columns) if one_sided else min(k, rows, columns)


# ── the eigenproblem ───────────────────────────────────────────────────────


def top_directions(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    k: int,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of ``P v = lambda C v`` with ``C = Q + ridge``.

    Reduced to a symmetric problem by whitening: with ``C = R R^T`` and
    ``v = R^-T w``,  ``P R^-T w = lambda R w  =>  (R^-1 P R^-T) w = lambda w``.
    The generalised eigenvectors are C-orthogonal rather than orthonormal, so a
    QR follows; it changes the basis but not the span, and the parametrisation
    depends only on the span because the core is free.
    """

    size = numerator.shape[0]
    P = numerator.double().cpu()
    Q = denominator.double().cpu()
    C = Q + delta * (torch.diagonal(Q).sum() / size) * torch.eye(size, dtype=torch.float64)
    # Grams are symmetric in exact arithmetic, but accumulation leaves them
    # asymmetric at roundoff level and both factorisations assume otherwise.
    C = 0.5 * (C + C.T)
    P = 0.5 * (P + P.T)
    try:
        R = torch.linalg.cholesky(C)
    except Exception:
        # Not positive definite at this ridge: floor the eigenvalues instead,
        # which regularises where it is needed rather than uniformly.
        values, vectors = torch.linalg.eigh(C)
        floor = max(float(values.max()) * 1e-10, 1e-30)
        C = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.T
        R = torch.linalg.cholesky(0.5 * (C + C.T))
    inverse = torch.linalg.solve_triangular(R, torch.eye(size, dtype=torch.float64), upper=False)
    reduced = inverse @ P @ inverse.T
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = torch.linalg.eigh(reduced)
    order = torch.argsort(values, descending=True)[:k]
    basis = torch.linalg.qr(inverse.T @ vectors[:, order])[0]
    return basis.float(), values[order].float()


def energy_ratio(
    basis: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor
) -> tuple[float, float, float]:
    """``(tr B^T P B / tr P) / (tr B^T Q B / tr Q)``: captured task energy over
    captured generic energy. About 1.0 for a basis that knows nothing."""

    B = basis.double().cpu()
    P = numerator.double().cpu()
    Q = denominator.double().cpu()
    own = float(torch.trace(B.T @ P @ B) / torch.trace(P))
    generic = float(torch.trace(B.T @ Q @ B) / torch.trace(Q))
    return own, generic, own / max(generic, 1e-30)


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """``||L^T R||_F^2 / k``, the mean squared principal cosine between two
    k-dimensional subspaces.

    Depends only on the spans, since ``||L^T R||_F^2 = ||P_L P_R||_F^2`` for the
    orthogonal projectors. For independent Haar-random subspaces its
    expectation is ``k/n``, which is the reference level: 0.083 at k=64 in 768
    dimensions, not zero.
    """

    return float((left.T @ right).pow(2).sum()) / left.shape[1]


# ── gradient statistics ────────────────────────────────────────────────────


@torch.enable_grad()
def accumulate_grams(
    *,
    model: nn.Module,
    matrices: Sequence[tuple[str, nn.Parameter]],
    loader: Iterable[Any],
    batches: int,
    kind: str,
    device: torch.device,
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """``E[G^T G]`` and ``E[G G^T]`` over ``batches`` aggregated-batch gradients.

    Per-sample Grams would be sharper and cost one backward per sample; the
    batch gradient is dominated by the mean direction, which is the one the
    optimizer actually follows.

    ``kind='task'``      : the task's own cross entropy.
    ``kind='reference'`` : ``r^T f(x)`` with a fresh random ``r`` per batch.
    Label-free, so the reference set needs no usable label space, and averaging
    over ``r`` probes the whole output Jacobian instead of one direction. That
    matters because a reference set such as ImageNet-21k has a label space the
    task head cannot score.
    """

    inputs = {
        name: torch.zeros(param.shape[1], param.shape[1], device=device, dtype=torch.float32)
        for name, param in matrices
    }
    outputs = {
        name: torch.zeros(param.shape[0], param.shape[0], device=device, dtype=torch.float32)
        for name, param in matrices
    }
    was_training = model.training
    model.eval()
    iterator = iter(loader)
    seen = 0
    for index in range(batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        output = model(images)
        if kind == "task":
            loss = F.cross_entropy(output, labels.to(device, non_blocking=True))
        else:
            normalized = output / (output.norm(dim=-1, keepdim=True) + 1e-8)
            probe = torch.randn(
                normalized.shape[-1], device=device, dtype=torch.float32, generator=generator
            )
            loss = (normalized.float() @ probe).sum()
        loss.backward()
        with torch.no_grad():
            for name, param in matrices:
                if param.grad is None:
                    continue
                gradient = param.grad.float()
                inputs[name] += gradient.T @ gradient
                outputs[name] += gradient @ gradient.T
        seen += 1
        if log is not None and (index + 1) % 10 == 0:
            log(f"{kind} statistics: {index + 1}/{batches} batches")
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    # These forwards ran with gradients enabled, so the runtime feature cache now
    # holds non-leaf tensors. Release them here, where they were created, rather
    # than leaving them for whatever runs next.
    from .strategies._subspace_parameterization import clear_runtime_feature_cache

    clear_runtime_feature_cache(model)
    for name in inputs:
        inputs[name] /= max(seen, 1)
        outputs[name] /= max(seen, 1)
    return inputs, outputs, seen


# ── assembly ───────────────────────────────────────────────────────────────


def solve_bases(
    *,
    matrices: Sequence[tuple[str, nn.Parameter]],
    task_inputs: Mapping[str, torch.Tensor],
    task_outputs: Mapping[str, torch.Tensor],
    reference_inputs: Mapping[str, torch.Tensor],
    reference_outputs: Mapping[str, torch.Tensor],
    k: int,
    delta: float,
    one_sided: bool,
    seed: int,
    no_reference: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor | None]], dict[str, dict[str, float]]]:
    """One eigenproblem per matrix per side, plus the diagnostics that say
    whether the result is selective at all."""

    bases: dict[str, dict[str, torch.Tensor | None]] = {}
    report: dict[str, dict[str, float]] = {}
    random = torch.Generator().manual_seed(seed + 2)
    for index, (name, param) in enumerate(matrices, 1):
        rows, columns = int(param.shape[0]), int(param.shape[1])
        rank = rank_for((rows, columns), k, one_sided)
        basis, _ = top_directions(task_inputs[name], reference_inputs[name], rank, delta)
        row: dict[str, float] = {"rank": float(rank)}
        row["own_in"], row["generic_in"], row["sel_in"] = energy_ratio(
            basis, task_inputs[name], reference_inputs[name]
        )
        reference = torch.linalg.qr(torch.randn(columns, rank, generator=random))[0]
        row["own_in_random"], _, row["sel_in_random"] = energy_ratio(
            reference, task_inputs[name], reference_inputs[name]
        )
        output_basis = None
        if not one_sided:
            output_basis, _ = top_directions(
                task_outputs[name], reference_outputs[name], rank, delta
            )
            row["own_out"], row["generic_out"], row["sel_out"] = energy_ratio(
                output_basis, task_outputs[name], reference_outputs[name]
            )
            reference = torch.linalg.qr(torch.randn(rows, rank, generator=random))[0]
            row["own_out_random"], _, row["sel_out_random"] = energy_ratio(
                reference, task_outputs[name], reference_outputs[name]
            )
        bases[name] = {"basis": basis.cpu(), "output_basis": None if output_basis is None else output_basis.cpu()}
        report[name] = row
        if log is not None and (index % 16 == 0 or index == len(matrices)):
            log(f"eigenproblems: {index}/{len(matrices)} matrices")

    def median(key: str) -> float:
        values = sorted(r[key] for r in report.values() if key in r and not math.isnan(r[key]))
        return values[len(values) // 2] if values else float("nan")

    if log is not None:
        if no_reference:
            # The generic energy of an orthonormal basis against the identity
            # is rank/size, so 'sel' reads as concentration of task energy
            # relative to a random subspace, not selectivity.
            log("no_reference ablation: the denominator is the identity, so 'sel' reads as")
            log("  concentration of task energy against a random basis, not selectivity")
        sides = [("input", "_in")] + ([] if one_sided else [("output", "_out")])
        log(f"basis report (medians over {len(matrices)} matrices, k={k}, delta={delta:g})")
        for label, suffix in sides:
            log(
                f"  {label:7s} own={median('own' + suffix):.4f} "
                f"generic={median('generic' + suffix):.2e} "
                f"sel={median('sel' + suffix):.2f} "
                f"| random own={median('own' + suffix + '_random'):.4f} "
                f"sel={median('sel' + suffix + '_random'):.2f}"
            )
        selectivity = median("sel_in")
        if no_reference:
            log(f"  ablation: median concentration {selectivity:.2f}x on the input side")
        elif selectivity <= 1.5:
            log(
                "  WARNING: not selective (sel ~ 1, no better than random). Raise "
                "delta, change the reference datasets, or use more batches."
            )
        else:
            log(f"  selective: median sel={selectivity:.2f} against about 1.0 for random")
    return bases, report


def cache_key(
    *,
    model_name: str,
    pretrained: str,
    task: str,
    k: int,
    delta: float,
    one_sided: bool,
    no_reference: bool,
    batches: int,
    reference_datasets: Sequence[str],
    seed: int,
) -> str:
    """A name carrying every knob that changes the bases, so two settings can
    never share a file."""

    payload = json.dumps(
        {
            "model": model_name,
            "pretrained": pretrained,
            "task": task,
            "k": k,
            "delta": delta,
            "one_sided": one_sided,
            "no_reference": no_reference,
            "batches": batches,
            "reference": list(reference_datasets),
            "seed": seed,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    parts = [task, f"k{k}", f"d{delta:g}"]
    if one_sided:
        parts.append("1s")
    if no_reference:
        parts.append("noref")
    return "subspace_" + "_".join(parts) + f"_{digest}.pt"


def load_cached(
    path: Path, matrices: Sequence[tuple[str, nn.Parameter]]
) -> dict[str, dict[str, torch.Tensor | None]] | None:
    """Return cached bases, or None when there is nothing usable on disk.

    A cache built for a different tensor set is refused rather than silently
    used for the matrices it happens to cover: a basis missing for one matrix
    would leave it untrained with nothing saying so.
    """

    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    bases = blob.get("bases") if isinstance(blob, Mapping) else None
    if not isinstance(bases, Mapping):
        raise ValueError(f"{path} is not a subspace basis file")
    missing = [name for name, _ in matrices if name not in bases]
    if missing:
        raise ValueError(
            f"{path} covers {len(bases)} matrices but this model needs "
            f"{len(matrices)}; first missing: {missing[0]}. Delete the file or "
            f"set strategy.params.subspace.rebuild=true."
        )
    return dict(bases)


def save_cached(
    path: Path,
    *,
    bases: Mapping[str, dict[str, torch.Tensor | None]],
    report: Mapping[str, dict[str, float]],
    meta: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({"bases": dict(bases), "report": dict(report), "meta": dict(meta)}, temporary)
    temporary.replace(path)
    with path.with_suffix(".json").open("w") as handle:
        json.dump({"meta": dict(meta), "report": dict(report)}, handle, indent=2)
