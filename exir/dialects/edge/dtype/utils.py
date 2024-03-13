# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
from typing import Any, List

import torch
from executorch.exir.dialects.edge.arg.model import BaseArg

from executorch.exir.dialects.edge.arg.type import ArgType


def extract_return_dtype(
    returns: Any, sample_returns: List[BaseArg]
) -> List[torch.dtype]:
    """Extract the dtype from a return value."""
    if not isinstance(returns, collections.abc.Sequence):
        returns = [returns]
    result = []
    for ret, sample in zip(returns, sample_returns):
        if sample.type == ArgType.TensorList or sample.type == ArgType.TensorOptList:
            # Assuming all tensors in tensor list has the same dtype, and we only add 1 dtype to result.
            assert (
                ret is not None
            ), f"Expecting non-None return value for {sample} but got None"
            result.append(ret.dtype)
            break
        elif sample.type == ArgType.Tensor or sample.type == ArgType.TensorOpt:
            assert (
                ret is not None
            ), f"Expecting non-None return value for {sample} but got None"
            result.append(ret.dtype)
    return result
