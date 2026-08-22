# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from executorch.exir._warnings import experimental
from torch import Tensor

@experimental("This API is experimental and subject to change without notice.")
class ExecuTorchSGD:
    """SGD Optimizer.

    .. warning::

        This API is experimental and subject to change without notice.
    """

    def step(self, named_gradients: Dict[str, Tensor]) -> None:
        """Take a step in the direction of the gradients."""
        ...

@experimental("This API is experimental and subject to change without notice.")
def get_sgd_optimizer(
    named_parameters: Dict[str, Tensor],
    lr: float,
    momentum: float = 0,
    dampening: float = 0,
    weight_decay: float = 0,
    nesterov: bool = False,
) -> ExecuTorchSGD:
    """Creates an sgd optimizer that operates on the passed in named_parameters according to the specified hyper parameters.

    .. warning::

        This API is experimental and subject to change without notice.
    ...
    """
    ...
