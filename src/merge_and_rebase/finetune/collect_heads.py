"""Rebuild heads.pt from per-task checkpoints.

``train_text`` accumulates the extracted heads of the tasks it trained and
writes them with

    torch.save(extracted_heads, out_dir / model_tag / "heads.pt")

which is a full overwrite. Training one task per job -- the only way to use
more than one GPU -- therefore leaves a heads.pt holding a single task, and the
last job to finish wins. The merge then has one head where it needs six, and
says so only when it fails to find the others.

The heads are recoverable because ``save_format: full`` stores the whole state
dict, classification head included. This reads each task's checkpoint, takes
the head parameters under the same canonical names ``_extract_task_head``
produces, and writes the combined file.

    python -m merge_and_rebase.finetune.collect_heads \\
        src/checkpoints/subspace_text/google-t5__t5-base \\
        --checkpoint subspace_best_ep.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

# The roots train_text recognises, in its preferred order.
HEAD_ROOTS = ("score", "classifier", "classification_head")


def canonical_param_name(name: str) -> str:
    """Mirror of ``train_text._canonical_param_name``.

    Copied rather than imported: that module pulls the whole data stack, and a
    tool for repairing a file on disk should run wherever the file is. The two
    are kept in step by a test that compares them when the stack is available.
    """

    out = str(name)
    if out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    return out.replace(".modules_to_save.default", "")


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}
        return {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    raise ValueError("unsupported checkpoint payload")


def extract_head(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The head parameters, under the canonical names the merge expects."""

    canonical = {canonical_param_name(k): v for k, v in state.items()}
    for root in HEAD_ROOTS:
        head = {
            name: tensor.detach().cpu().clone()
            for name, tensor in canonical.items()
            if name == root or name.startswith(root + ".")
        }
        if head:
            return head
    raise RuntimeError(
        f"no classification head found; looked for {', '.join(HEAD_ROOTS)} among "
        f"{sorted(canonical)[:6]}"
    )


def collect_sidecars(root: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Heads from the per-task files train_text writes beside heads.pt."""

    directory = root / "heads"
    if not directory.is_dir():
        return {}
    heads: dict[str, dict[str, torch.Tensor]] = {}
    for path in sorted(directory.glob("*.pt")):
        try:
            head = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if isinstance(head, dict) and head:
            heads[path.stem] = head
    return heads


def collect(
    root: Path, checkpoint: str, tasks: list[str] | None = None
) -> dict[str, dict[str, torch.Tensor]]:
    """One head per task, from the sidecars when present, else the checkpoints.

    Sidecars are what current runs write and are cheap to read; the checkpoint
    path recovers runs made before they existed.
    """

    sidecars = collect_sidecars(root)
    candidates = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    heads: dict[str, dict[str, torch.Tensor]] = {}
    missing: list[str] = []
    for directory in candidates:
        task = directory.name
        if tasks is not None and task not in tasks:
            continue
        if task in sidecars:
            heads[task] = sidecars[task]
            continue
        path = directory / checkpoint
        if not path.exists():
            missing.append(task)
            continue
        heads[task] = extract_head(_state_dict(torch.load(path, map_location="cpu", weights_only=False)))
    if tasks is not None:
        absent = [t for t in tasks if t not in heads]
        if absent:
            raise RuntimeError(
                f"no {checkpoint} for: {', '.join(absent)}. Those experts are "
                f"missing or were saved under another name, and a merge with a "
                f"partial head set would be a wrong result, not a partial one."
            )
    # Sidecars for tasks with no directory of their own still count.
    for task, head in sidecars.items():
        if task not in heads and (tasks is None or task in tasks):
            heads[task] = head
    if not heads:
        raise RuntimeError(
            f"no head found under {root}: neither heads/<task>.pt nor "
            f"<task>/{checkpoint}"
        )
    if missing:
        print(f"  skipped (no {checkpoint}): {', '.join(missing)}")
    return heads


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory holding one subdirectory per task")
    parser.add_argument("--checkpoint", default="subspace_best_ep.pt")
    parser.add_argument("--tasks", help="comma-separated; all task directories by default")
    parser.add_argument("--out", type=Path, help="default: <root>/heads.pt")
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    heads = collect(args.root, args.checkpoint, tasks)
    out = args.out or (args.root / "heads.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads, out)
    print(f"  {len(heads)} heads -> {out}")
    for task, head in sorted(heads.items()):
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(head.items()))
        print(f"    {task:10s} {shapes}")


if __name__ == "__main__":
    main()
