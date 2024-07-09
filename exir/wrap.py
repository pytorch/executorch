# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Helper functions for constructing a "leaf function" in FX graph. A "leaf
function" will be preserved as a call node in the the graph instead of
being traced through.
"""

import torch
from executorch.exir.tracer import PythonTensor, unwrap_functional

# pyre-fixme[21]: Could not find module `torch._C._functorch`.
from torch._C._functorch import (  # @manual=//caffe2/functorch:functorch"
    is_functionaltensor,
)

from torch._functorch.eager_transforms import _assert_wrapped_functional  # pyre-ignore


def update_with_proxy(t: torch.Tensor, proxy: torch.fx.Proxy) -> torch.Tensor:
    unwrapped = unwrap_functional(t)
    assert isinstance(unwrapped, PythonTensor)
    unwrapped.update_proxy(proxy)
    if is_functionaltensor(t):  # type: ignore
        _assert_wrapped_functional(unwrapped, t)
    return t
