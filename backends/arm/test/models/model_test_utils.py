# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch


def to_bfloat16(
    model: torch.nn.Module, inputs: tuple[Any, ...]
) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    return model.to(torch.bfloat16), tuple(
        (
            x.to(torch.bfloat16)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else x
        )
        for x in inputs
    )
