# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple, Union

import torch
from executorch.exir.delegate import executorch_call_delegate
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


DELEGATE_SCRATCH_SPECS_META_KEY = "delegate_scratch_specs"


def delegate_scratch_specs(node: torch.fx.Node) -> List[TensorSpec]:
    """Returns the scratch buffers a delegate call needs while it executes.

    ``DelegateScratchSpecPass`` puts them here, the memory planner places them,
    and the emitter serializes the result onto the delegate call. They are not
    values the node produces, so they are kept out of ``meta["spec"]``.

    The target check is what keeps the key from meaning anything anywhere else.
    Any pass can write metadata, and only a delegate call is ever asked for its
    scratch, so a stray key elsewhere would otherwise be planned into the arena
    and never emitted.
    """
    if node.target is not executorch_call_delegate:
        return []
    return node.meta.get(DELEGATE_SCRATCH_SPECS_META_KEY, [])


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
