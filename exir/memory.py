# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple, Union

import torch
from executorch.exir.sym_util import eval_shape
from executorch.exir.tensor import TensorSpec

from torch.utils import _pytree as pytree
from typing_extensions import TypeAlias

TensorAllocSpec: TypeAlias = Tuple[Tuple[int], torch.dtype]
AllocSpec: TypeAlias = Union[
    TensorAllocSpec,
    List[TensorAllocSpec],
]


def alloc(spec: AllocSpec) -> pytree.PyTree:
    if isinstance(spec, list):
        return [alloc(s) for s in spec]

    shape, dtype = spec
    # evaluate the shape to int so we can run the traced module
    # in python for testing
    shape = eval_shape(shape)
    return torch.empty(shape, dtype=dtype)


def free(spec: TensorSpec) -> None:
    """
    The function is nop. The major purpose is to put it in the Fx IR.
    E.g., it can be the target of call_function node.
    """
    pass


def view(base: torch.Tensor, size: List[int]) -> torch.Tensor:
    """
    This function mimics torch.ops.aten.view.default.

    It is used to elide view_copy nodes.
    """
    return base.view(size)
