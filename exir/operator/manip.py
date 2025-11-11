# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains APIs to manipulate ops.
"""
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from executorch.exir.tensor import TensorSpec


@dataclass
class ScratchTensorMetadata:
    dtype: torch.dtype
    shape: torch.Size
    layout: torch.layout = torch.strided
    device: torch.device = torch.device("cpu")
    is_sparse: bool = False


ScratchCallableType = Callable[..., Dict[str, ScratchTensorMetadata]]


def attach_get_scratch_metas_fn(
    out_variant: torch._ops.OpOverload,
) -> Callable[[ScratchCallableType], ScratchCallableType]:
    """
    Apply this decorator to the get_scratch_metas methods for `out_variant` op.
    The decorator will do the job to attach the get_scratch_metas method
    to the out variant op.

    The get_scratch_metas method has the same signature as the out variant op.
    There are 2 difference though:
    - the Tensor input arguments are all replaced with TensorSpec
    - the output is a dictionary of ScratchTensorMetadata
    """

    def to_tensor_spec(meta: ScratchTensorMetadata) -> TensorSpec:
        return TensorSpec(
            const=False,
            requires_grad=False,
            # fields copy from ScratchTensorMetadata
            dtype=meta.dtype,
            shape=meta.shape,
            layout=meta.layout,
            is_sparse=meta.is_sparse,
        )

    def adapt_return_value(
        get_scratch_metas_fn: ScratchCallableType,
    ) -> Callable[..., Dict[str, TensorSpec]]:
        """
        Adapt return value from a ScratchTensorMetadata to a TensorSpec
        """

        def wrapper(*args: TensorSpec, **kwargs: TensorSpec) -> Dict[str, TensorSpec]:
            meta_dict = get_scratch_metas_fn(*args, **kwargs)
            return {k: to_tensor_spec(v) for k, v in meta_dict.items()}

        return wrapper

    def wrapper(get_scratch_metas_fn: ScratchCallableType) -> ScratchCallableType:
        # pyre-fixme[16]: `OpOverload` has no attribute `get_scratch_metas`.
        out_variant.get_scratch_metas = adapt_return_value(get_scratch_metas_fn)
        return get_scratch_metas_fn

    return wrapper


# pyre-ignore
def attach_calculate_upper_bound_shape_fn(func_op: torch._ops.OpOverload):
    """
    The input is the OpOverload for the functional op.
    """

    # pyre-ignore
    def wrapper(calculate_upper_bound_shape_fn):
        # pyre-fixme[16]: `OpOverload` has no attribute `calculate_upper_bound_shape`.
        func_op.calculate_upper_bound_shape = calculate_upper_bound_shape_fn
        return calculate_upper_bound_shape_fn

    return wrapper
