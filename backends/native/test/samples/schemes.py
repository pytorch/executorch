# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization schemes. Every op is recorded under every scheme, keeping the
cases the scheme fits."""

from dataclasses import dataclass

import torch
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from torchao.utils import TorchAOBaseTensor


class Scheme:
    """No quantization."""

    def fits(self, model: torch.nn.Module) -> bool:
        return True

    def quantize(self, model: torch.nn.Module) -> None:
        pass

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        """A state tensor of a quantized model as a plain tensor."""
        return value


def _fits(model: torch.nn.Module, group: int | None) -> bool:
    """Whether `model` has an nn.Linear and every one's in_features split into
    groups of `group`, or form one group when it is None."""
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    for m in linears:
        size = m.in_features if group is None else group
        # torchao reduces a one-element-group weight to a single scale.
        if size < 2 or m.in_features % size:
            return False
    return bool(linears)


@dataclass(frozen=True)
class Torchao(Scheme):
    """A torchao config applied to every nn.Linear."""

    config: AOBaseConfig

    def fits(self, model: torch.nn.Module) -> bool:
        granularity = (
            self.config.granularity
            if isinstance(self.config, IntxWeightOnlyConfig)
            else self.config.weight_granularity
        )
        return _fits(
            model, granularity.group_size if isinstance(granularity, PerGroup) else None
        )

    def quantize(self, model: torch.nn.Module) -> None:
        quantize_(model, self.config)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value.dequantize() if isinstance(value, TorchAOBaseTensor) else value


def schemes() -> dict[str, Scheme]:
    return {
        "float": Scheme(),
        "int4wo_g32": Torchao(
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32))
        ),
        "int4wo_g128": Torchao(
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(128))
        ),
        "8da4w_g32": Torchao(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(32)
            )
        ),
        "8da4w_g128": Torchao(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(128)
            )
        ),
    }
