"""Subspace-constrained fine-tuning for text models.

Same method as the vision path -- every eligible matrix is reparameterised as
``W = W_0 + U C V^T`` with ``U`` and ``V`` fixed before training -- with three
differences that a text model forces, each of which changes what is measured
and so is spelled out here rather than buried.

The probe is not on the logits
------------------------------
The vision reference statistics differentiate ``r^T f(x)`` on a 512-dimensional
embedding. A sequence-classification head emits three logits, and a random
direction in three dimensions samples the Jacobian far too coarsely to say
anything about which directions the model uses generically. The probe therefore
runs on the last hidden state, mean-pooled over the non-padding tokens, which
is the closest analogue of the vision embedding.

The vocabulary embedding stays out
----------------------------------
``shared.weight`` is 32128x768, shared by every task, and on T5 it is tied to
the output head. It belongs with the convolution stem and the LayerNorms in the
frozen remainder, not with the attention and feed-forward matrices: those are
what the SVD-based mergers act on, and what the method is about.

The classification head trains normally
---------------------------------------
Unlike CLIP's frozen zero-shot head, this one is randomly initialised: there is
no pre-trained ``W_0`` near which to stay, so constraining it to a subspace
around noise would be meaningless. The repository also extracts heads per task
rather than merging them, so the head never enters a task vector.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import subspace_bases as sb
from .strategies._subspace_parameterization import (
    bind_subspace_parameterization,
    clear_runtime_feature_cache,
)

# Shared across every expert, or not a transformer weight matrix at all.
_EXCLUDE = re.compile(
    r"(^|\.)(shared|embed_tokens|lm_head|wte|wpe|word_embeddings|position_embeddings"
    r"|token_type_embeddings|relative_attention_bias)(\.|$)"
)
# The task head: randomly initialised, trained normally, never merged.
_HEAD = re.compile(
    r"(^|\.)(classification_head|classifier|score|qa_outputs|pooler)(\.|$)"
)

ALLOWED: dict[str, Any] = {
    "k": 64,
    "delta": 1e-2,
    "batches": 30,
    "reference_batches": 0,
    "reference_dataset": "wikitext",
    "reference_config": "wikitext-103-raw-v1",
    "reference_split": "train",
    "reference_size": 4096,
    "max_length": 128,
    "one_sided": False,
    "no_reference": False,
    "cache_dir": None,
    "rebuild": False,
}


def validate(settings: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(settings or {})
    unknown = set(values) - set(ALLOWED)
    if unknown:
        raise ValueError(
            f"unsupported strategy.params.subspace options: {sorted(unknown)}. "
            f"Available: {sorted(ALLOWED)}"
        )
    out = dict(ALLOWED)
    out.update(values)
    out["k"] = int(out["k"])
    if out["k"] < 1:
        raise ValueError("strategy.params.subspace.k must be positive")
    out["delta"] = float(out["delta"])
    if out["delta"] < 0:
        raise ValueError("strategy.params.subspace.delta must not be negative")
    out["batches"] = int(out["batches"])
    if out["batches"] < 1:
        raise ValueError("strategy.params.subspace.batches must be positive")
    out["reference_batches"] = int(out["reference_batches"])
    out["reference_size"] = int(out["reference_size"])
    out["max_length"] = int(out["max_length"])
    out["one_sided"] = bool(out["one_sided"])
    out["no_reference"] = bool(out["no_reference"])
    out["rebuild"] = bool(out["rebuild"])
    return out


def head_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """The task head, which trains normally."""

    return [(name, p) for name, p in model.named_parameters() if _HEAD.search(name)]


def _embedding_parameter_ids(model: nn.Module) -> set[int]:
    """Parameters owned by an embedding module, found by type rather than name.

    The name patterns cover the architectures here, but they are a convention:
    an embedding reached under an unexpected attribute would slip through and
    end up in the subspace, where a 32128x768 table shared by every expert does
    not belong. Module type is exact.
    """

    owned: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            owned.update(id(p) for p in module.parameters(recurse=False))
    return owned


def eligible_matrices(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """The 2-D transformer weights: attention and feed-forward projections.

    Excludes the vocabulary and position embeddings, the relative-attention
    bias and the task head. On ``t5-v1_1-base`` this leaves 12 encoder blocks
    and 12 decoder blocks of {q, k, v, o} plus their feed-forward matrices.
    """

    embeddings = _embedding_parameter_ids(model)
    selected = [
        (name, p)
        for name, p in model.named_parameters()
        if p.dim() == 2
        and id(p) not in embeddings
        and not _EXCLUDE.search(name)
        and not _HEAD.search(name)
    ]
    if not selected:
        raise RuntimeError(
            "the text subspace selected zero matrices; the exclusion patterns "
            "may not match this architecture's parameter names"
        )
    return selected


# ── objectives ─────────────────────────────────────────────────────────────


def _forward_kwargs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    kwargs = {"input_ids": batch["input_ids"].to(device, non_blocking=True)}
    mask = batch.get("attention_mask")
    if mask is not None:
        kwargs["attention_mask"] = mask.to(device, non_blocking=True)
    return kwargs


def task_objective(model: nn.Module, batch: dict[str, Any], device, generator=None):
    """The task's own loss, exactly as the training loop computes it."""

    del generator
    kwargs = _forward_kwargs(batch, device)
    labels = batch["labels"].to(device, non_blocking=True).long()
    outputs = model(**kwargs, labels=labels)
    loss = outputs.loss
    if loss is None:
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
    return loss


# Decoder first, deliberately. An encoder-decoder's encoder output does not
# depend on the decoder weights, so probing it would leave every decoder matrix
# with no reference signal at all -- a denominator of pure ridge, and a basis
# chosen from the task Grams alone on half the model.
_HIDDEN_ATTRS = ("decoder_hidden_states", "hidden_states", "encoder_hidden_states")
_LAST_HIDDEN_ATTRS = ("last_hidden_state", "encoder_last_hidden_state")


def _last_hidden_state(outputs: Any) -> tuple[torch.Tensor, str]:
    """The deepest representation that depends on as much of the model as
    possible, with the name of where it came from."""

    for attribute in _HIDDEN_ATTRS:
        states = getattr(outputs, attribute, None)
        if states:
            return states[-1], attribute
    for attribute in _LAST_HIDDEN_ATTRS:
        state = getattr(outputs, attribute, None)
        if state is not None:
            return state, attribute
    raise RuntimeError(
        "the model exposes no hidden states, so the reference probe has nothing "
        "to differentiate. Tried: " + ", ".join(_HIDDEN_ATTRS + _LAST_HIDDEN_ATTRS)
    )


def reference_objective(model: nn.Module, batch: dict[str, Any], device, generator=None):
    """``r^T h(x)`` on the mean-pooled last hidden state.

    Label-free, so a generic corpus needs no annotation. The input mask is
    applied only when it matches the sequence length: on an encoder-decoder the
    representation that reaches here is the decoder's, whose length is unrelated
    to the input, and masking it with the encoder's mask would be meaningless.
    """

    kwargs = _forward_kwargs(batch, device)
    outputs = model(**kwargs, output_hidden_states=True)
    last, _source = _last_hidden_state(outputs)
    mask = kwargs.get("attention_mask")
    if mask is not None and mask.shape[1] == last.shape[1]:
        weights = mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    else:
        pooled = last.mean(dim=1)
    pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
    probe = torch.randn(
        pooled.shape[-1], device=device, dtype=torch.float32, generator=generator
    )
    return (pooled.float() @ probe).sum()


# ── the reference corpus ───────────────────────────────────────────────────


def reference_loader(
    settings: dict[str, Any],
    tokenizer: Any,
    batch_size: int,
    seed: int,
    log: Callable[[str], None],
):
    """A loader of generic text, tokenised with the model's own tokenizer.

    Deliberately independent of the NLI loaders: the reference set must not be
    one of the tasks being merged, and the probe needs no labels, so going
    through a task loader would only impose a label space that is never read.
    """

    from datasets import load_dataset

    name = str(settings["reference_dataset"])
    config = settings["reference_config"] or None
    split = str(settings["reference_split"])
    limit = int(settings["reference_size"])
    # Recent `datasets` rejects a bare repository id, so a configured short
    # name is resolved the same way the NLI loaders resolve theirs.
    from ..data.text_loaders import canonical_dataset_paths

    data = None
    errors: list[str] = []
    for candidate in canonical_dataset_paths(name):
        try:
            data = load_dataset(candidate, config, split=split)
            name = candidate
            break
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if data is None:
        raise RuntimeError(
            "could not load the reference corpus. Tried:\n  " + "\n  ".join(errors)
        )
    column = next(
        (c for c in ("text", "sentence", "content", "document") if c in data.column_names),
        None,
    )
    if column is None:
        raise RuntimeError(
            f"{name} exposes no text column; found {data.column_names}"
        )
    if limit and limit < len(data):
        data = data.shuffle(seed=seed).select(range(limit))
    # Blank lines are a third of wikitext and carry no signal; a probe batch of
    # padding would contribute a gradient of almost nothing and dilute the
    # estimate.
    rows = [t for t in data[column] if isinstance(t, str) and len(t.strip()) > 32]
    if not rows:
        raise RuntimeError(f"{name} yielded no usable text rows")
    log(f"reference corpus {name}: {len(rows)} rows from split {split!r}")

    max_length = int(settings["max_length"])

    class _Corpus(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> str:
            return rows[index]

    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    return torch.utils.data.DataLoader(
        _Corpus(), batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )


# ── entry point ────────────────────────────────────────────────────────────


def prepare(
    *,
    model: nn.Module,
    settings: dict[str, Any] | None,
    task: str,
    train_loader: Iterable[Any],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    seed: int,
    model_tag: str,
    cache_root: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Build or load the bases, bind the parameterisation, set requires_grad.

    Leaves the model with exactly the cores and the task head trainable, so the
    caller's generic ``[p for p in model.parameters() if p.requires_grad]``
    collects the right set without knowing about any of this.
    """

    values = validate(settings)
    matrices = eligible_matrices(model)
    head = head_parameters(model)
    shapes: dict[tuple[int, int], int] = {}
    for _, p in matrices:
        shapes[tuple(p.shape)] = shapes.get(tuple(p.shape), 0) + 1
    log(
        f"{len(matrices)} eligible matrices, k={values['k']}"
        + (" (one-sided)" if values["one_sided"] else " (both sides)")
        + f"; head tensors trained normally: {len(head)}"
    )
    log("  " + ", ".join(f"{c}x({s[0]}x{s[1]})" for s, c in sorted(shapes.items())))

    path = Path(values["cache_dir"] or (cache_root / "subspace_bases")) / sb.cache_key(
        model_name=model_tag,
        pretrained="hf",
        task=task,
        k=values["k"],
        delta=values["delta"],
        one_sided=values["one_sided"],
        no_reference=values["no_reference"],
        batches=values["batches"],
        reference_datasets=(
            () if values["no_reference"]
            else (f"{values['reference_dataset']}/{values['reference_config']}",)
        ),
        seed=seed,
    )
    bases = None if values["rebuild"] else sb.load_cached(path, matrices)
    if bases is not None:
        log(f"bases loaded from {path}")
    else:
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        for _, p in matrices:
            p.requires_grad_(True)

        log(f"task gradients: {values['batches']} batches")
        task_inputs, task_outputs, seen = sb.accumulate_grams(
            model=model, matrices=matrices, loader=train_loader,
            batches=values["batches"], kind="task", device=device, log=log,
            objective=task_objective,
        )
        dead = [n for n, _ in matrices if float(torch.diagonal(task_inputs[n]).sum()) <= 0]
        if dead:
            raise RuntimeError(
                f"{len(dead)} matrix/matrices received no gradient from the task "
                f"loss: {dead[:8]}. On an encoder-decoder, the decoder is not on "
                f"the forward path of every head; exclude it or use a task that "
                f"drives it."
            )

        reference_batches = values["reference_batches"] or values["batches"]
        if values["no_reference"]:
            log("no_reference: the denominator is the identity")
            reference_inputs = {n: torch.eye(p.shape[1], device=device) for n, p in matrices}
            reference_outputs = {n: torch.eye(p.shape[0], device=device) for n, p in matrices}
            reference_batches = 0
        else:
            loader = reference_loader(values, tokenizer, batch_size, seed, log)
            log(f"reference gradients: {reference_batches} batches")
            generator = torch.Generator(device=device).manual_seed(seed + 2)
            reference_inputs, reference_outputs, _ = sb.accumulate_grams(
                model=model, matrices=matrices, loader=loader,
                batches=reference_batches, kind="reference", device=device,
                generator=generator, log=log, objective=reference_objective,
            )

        if not values["no_reference"]:
            # A matrix with a zero reference Gram would be chosen from the task
            # statistics alone, with the ridge as its only denominator: not the
            # method, and invisible in the result.
            blind = [
                n for n, _ in matrices
                if float(torch.diagonal(reference_inputs[n]).sum()) <= 0
            ]
            if blind:
                raise RuntimeError(
                    f"{len(blind)} matrix/matrices received no gradient from the "
                    f"reference probe, so they have no denominator: {blind[:8]}. "
                    f"The probe must depend on every matrix in the subspace."
                )

        log("solving the eigenproblems (fp64, on CPU)")
        bases, report = sb.solve_bases(
            matrices=matrices,
            task_inputs=task_inputs, task_outputs=task_outputs,
            reference_inputs=reference_inputs, reference_outputs=reference_outputs,
            k=values["k"], delta=values["delta"], one_sided=values["one_sided"],
            seed=seed, no_reference=values["no_reference"], log=log,
        )
        sb.save_cached(
            path, bases=bases, report=report,
            meta={
                "task": task, "model": model_tag, "k": values["k"],
                "delta": values["delta"], "one_sided": values["one_sided"],
                "no_reference": values["no_reference"],
                "task_batches": seen, "reference_batches": reference_batches,
                "reference_dataset": None if values["no_reference"] else values["reference_dataset"],
                "reference_config": None if values["no_reference"] else values["reference_config"],
                "probe": "mean-pooled last hidden state",
                "batch_size": batch_size, "seed": seed,
                "matrices": [n for n, _ in matrices],
            },
        )
        log(f"bases saved to {path}")
        del task_inputs, task_outputs, reference_inputs, reference_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    before = {n: p.detach().clone() for n, p in matrices}
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    named = list(model.named_parameters())
    clear_runtime_feature_cache(model)
    cores = bind_subspace_parameterization(
        model=model, named_params=named,
        bases={n: bases[n] for n, _ in matrices}, device=device,
    )
    # The head is new weights, not a displacement from anything, so it trains
    # normally. It is extracted per task and never merged.
    for _, p in head_parameters(model):
        p.requires_grad_(True)

    current = model._current_param_map()
    drift = max(
        float((current[n].detach().to(w.device) - w).abs().max()) for n, w in before.items()
    )
    log(
        f"{sum(c.numel() for c in cores)} core parameters, "
        f"{sum(p.numel() for _, p in head_parameters(model))} head parameters; "
        f"max |W - W_0| at init = {drift:.3e} {'[ok]' if drift < 1e-5 else '[FAIL]'}"
    )
    if drift >= 1e-5:
        raise RuntimeError(
            "the subspace parameterisation does not start at the pre-trained "
            "weights; refusing to train"
        )
    return {
        "subspace_matrices": len(matrices),
        "subspace_k": values["k"],
        "subspace_delta": values["delta"],
        "subspace_one_sided": values["one_sided"],
        "subspace_no_reference": values["no_reference"],
        "subspace_cores": int(sum(c.numel() for c in cores)),
        "subspace_head_params": int(sum(p.numel() for _, p in head_parameters(model))),
    }
