# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed input-channel 2:4 masks for convolution weight pruning."""

from collections.abc import Sequence
from fnmatch import fnmatchcase

import torch


class TwoOfFourPruner:
    """Keep two weights per four input channels in eager, PTQ, or QAT.

    Groups are independent for each output channel and kernel position. Masks
    are selected by magnitude once and stored outside the model. This helper
    zeros masked weights in place; call apply_ after optimizer updates to
    restore those zeros. Eager evaluation and export see ordinary weight
    tensors, without pruning hooks, parametrizations or runtime masking
    operations.

    """

    def __init__(
        self, model: torch.nn.Module, layer_patterns: Sequence[str] = ()
    ) -> None:
        """Select convolution modules, create masks, and zero masked weights.

        Args:
            model (torch.nn.Module): Model copy to prune for eager evaluation
                or PT2E preparation.
            layer_patterns (Sequence[str]): Shell-style module-name patterns.
                Every pattern must match. Only Conv2d weights with a multiple
                of four input channels per group are supported.

        """
        modules = dict(model.named_modules(remove_duplicate=False))
        selected: set[str] = set()
        for pattern in layer_patterns:
            matches = {name for name in modules if fnmatchcase(name, pattern)}
            if not matches:
                raise ValueError(f"Pruning pattern matched no layers: {pattern!r}")
            selected.update(matches)
        self.weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        seen: set[int] = set()
        for name in sorted(selected):
            module = modules[name]
            if not isinstance(module, torch.nn.Conv2d):
                raise ValueError(f"2:4 pruning requires a Conv2d layer: {name}")
            weight = module.weight
            if (
                not weight.is_floating_point()
                or weight.shape[1] % 4
                or not torch.isfinite(weight).all()
            ):
                raise ValueError(
                    f"2:4 pruning requires finite weights and a multiple of "
                    f"four input channels per group: {name}"
                )
            if id(weight) in seen:
                continue
            seen.add(id(weight))
            groups = (
                weight.detach()
                .permute(0, 2, 3, 1)
                .reshape(weight.shape[0], weight.shape[2], weight.shape[3], -1, 4)
            )
            indices = groups.abs().argsort(dim=-1, descending=True, stable=True)
            mask = torch.zeros_like(groups, dtype=torch.bool)
            mask.scatter_(-1, indices[..., :2], True)
            mask = mask.flatten(-2).permute(0, 3, 1, 2).contiguous()
            self.weights[name] = (weight, mask)
        self.apply_(model)
        if selected:
            print(f"2:4 pruned layers: {', '.join(sorted(selected))}")

    @torch.no_grad()
    def apply_(self, model: torch.nn.Module) -> None:
        """Zero masked weights in place, including after optimizer updates.

        Args:
            model (torch.nn.Module): Eager or prepared PT2E model retaining
                the original parameter objects, even if names have changed.

        """
        if not self.weights:
            return
        parameter_ids = {id(p) for p in model.parameters()}
        for name, (weight, mask) in self.weights.items():
            if id(weight) not in parameter_ids:
                raise ValueError(f"Quantization replaced the pruned weight: {name}")
            weight.masked_fill_(~mask.to(device=weight.device), 0)
