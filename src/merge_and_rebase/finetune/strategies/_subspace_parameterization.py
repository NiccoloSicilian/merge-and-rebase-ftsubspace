"""W = W_0 + U C V^T, bound the same way as the plain delta parameterisation.

This mirrors ``_delta_parameterization`` -- a deep-copied reference module
driven by ``functional_call`` over a materialised parameter map -- so the
forward path, the runtime feature caches and ``_materialized_state_dict`` all
behave identically. The only difference is what the delta is made of: instead
of one free tensor per target matrix there is a small core ``C``, with ``U`` and
``V`` frozen buffers.

The cores start at zero, so the model begins exactly at the pre-trained
weights: the subspace changes the hypothesis space, not the starting point.
Because ``U`` and ``V`` have orthonormal columns, ``||Delta||_F = ||C||_F``,
which is a free self-check during training.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MethodType

import torch
import torch.nn as nn
from torch.func import functional_call

from ._delta_parameterization import _sync_runtime_feature_cache


class _SubspaceModule(nn.Module):
    """The trainable cores, plus the frozen bases they are sandwiched between."""

    def __init__(
        self,
        names: list[str],
        cores: list[nn.Parameter],
        bases: list[torch.Tensor],
        output_bases: list[torch.Tensor | None],
    ) -> None:
        super().__init__()
        self.names = tuple(names)
        self.params = nn.ParameterList(cores)
        # Buffers, not parameters: they must move with the module and be saved
        # nowhere, since the checkpoint is the materialised weight.
        for index, basis in enumerate(bases):
            self.register_buffer(f"basis_{index}", basis, persistent=False)
        for index, basis in enumerate(output_bases):
            if basis is not None:
                self.register_buffer(f"output_basis_{index}", basis, persistent=False)
        self._two_sided = tuple(basis is not None for basis in output_bases)

    def delta(self, index: int) -> torch.Tensor:
        core = self.params[index]
        basis = getattr(self, f"basis_{index}")
        # fp32 regardless of any surrounding autocast: the delta is a small
        # correction to a large weight and bf16 would quantise it away.
        with torch.autocast(device_type=core.device.type, enabled=False):
            value = core.float() @ basis.float().T
            if self._two_sided[index]:
                output_basis = getattr(self, f"output_basis_{index}")
                value = output_basis.float() @ core.float() @ basis.float().T
        return value

    def by_name(self) -> dict[str, torch.Tensor]:
        return {name: self.delta(index) for index, name in enumerate(self.names)}

    def core_norm(self) -> float:
        total = 0.0
        for core in self.params:
            total += float(core.detach().float().pow(2).sum())
        return float(total**0.5)


def _materialized_param_map(
    *,
    named_params: list[tuple[str, nn.Parameter]],
    base_params: Mapping[str, torch.Tensor],
    subspace_module: _SubspaceModule,
) -> dict[str, torch.Tensor]:
    deltas = subspace_module.by_name()
    param_map: dict[str, torch.Tensor] = {}
    for name, param in named_params:
        if name in deltas:
            delta = deltas[name]
            base = base_params[name].to(device=delta.device, dtype=delta.dtype)
            param_map[name] = base + delta
        else:
            param_map[name] = param
    return param_map


def _materialized_state_dict(
    *,
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    base_params: Mapping[str, torch.Tensor],
    subspace_module: _SubspaceModule,
) -> dict[str, torch.Tensor]:
    """The plain state dict with the delta folded in.

    Folding before saving is not optional: an expert is consumed downstream as
    an ordinary state dict and the task vector is a difference of tensors by
    name, so a checkpoint that kept the cores separate would lose every matrix
    the subspace touched.
    """

    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key in [key for key in state if key.startswith("_subspace_module.")]:
        state.pop(key, None)
    deltas = subspace_module.by_name()
    for name, _param in named_params:
        if name not in deltas or name not in state or name not in base_params:
            continue
        delta = deltas[name]
        base = base_params[name].to(device=delta.device, dtype=delta.dtype)
        state[name] = (base + delta).detach().cpu()
    return state


def bind_subspace_parameterization(
    *,
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    bases: Mapping[str, Mapping[str, torch.Tensor | None]],
    device: torch.device,
) -> list[nn.Parameter]:
    """Install the cores and route the forward through the materialised weights.

    Returns the cores, which are the only tensors an optimizer should see.
    """

    if not bases:
        raise RuntimeError("the subspace parameterisation received no bases")

    target_names = set(bases)
    base_params = {
        name: param.detach().to(device=device).clone()
        for name, param in named_params
        if name in target_names
    }
    missing = target_names - set(base_params)
    if missing:
        raise RuntimeError(
            f"bases were built for parameters this model does not have: {sorted(missing)[:5]}"
        )

    names: list[str] = []
    cores: list[nn.Parameter] = []
    input_bases: list[torch.Tensor] = []
    output_bases: list[torch.Tensor | None] = []
    for name, param in named_params:
        if name not in target_names:
            continue
        entry = bases[name]
        basis = entry["basis"].to(device=device, dtype=torch.float32)
        output_basis = entry.get("output_basis")
        output_basis = (
            None if output_basis is None else output_basis.to(device=device, dtype=torch.float32)
        )
        if basis.shape[0] != param.shape[1]:
            raise RuntimeError(
                f"{name}: basis has {basis.shape[0]} rows but the weight has "
                f"{param.shape[1]} columns"
            )
        if output_basis is not None and output_basis.shape[0] != param.shape[0]:
            raise RuntimeError(
                f"{name}: output basis has {output_basis.shape[0]} rows but the "
                f"weight has {param.shape[0]}"
            )
        rows = param.shape[0] if output_basis is None else output_basis.shape[1]
        param.requires_grad_(False)
        names.append(name)
        cores.append(nn.Parameter(torch.zeros(rows, basis.shape[1], dtype=torch.float32, device=device)))
        input_bases.append(basis)
        output_bases.append(output_basis)

    ref_model = deepcopy(model).to(device)
    subspace_module = _SubspaceModule(names, cores, input_bases, output_bases).to(device)

    def _current_param_map() -> dict[str, torch.Tensor]:
        return _materialized_param_map(
            named_params=named_params, base_params=base_params, subspace_module=subspace_module
        )

    def _materialized_state() -> dict[str, torch.Tensor]:
        return _materialized_state_dict(
            model=model,
            named_params=named_params,
            base_params=base_params,
            subspace_module=subspace_module,
        )

    def _subspace_forward(self, *args, **kwargs):
        out = functional_call(ref_model, _current_param_map(), args=args, kwargs=kwargs, strict=False)
        _sync_runtime_feature_cache(target=self, source=ref_model)
        return out

    model._subspace_module = subspace_module  # type: ignore[attr-defined]
    model._current_param_map = _current_param_map  # type: ignore[attr-defined]
    model._materialized_state_dict = _materialized_state  # type: ignore[attr-defined]
    model.forward = MethodType(_subspace_forward, model)  # type: ignore[method-assign]
    return list(subspace_module.params)
