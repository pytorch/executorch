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

from torch._functorch.eager_transforms import _assert_wrapped_functional


def update_with_proxy(t: torch.Tensor, proxy: torch.fx.Proxy) -> torch.Tensor:
    unwrapped = unwrap_functional(t)
    assert isinstance(unwrapped, PythonTensor)
    unwrapped.update_proxy(proxy)
    if is_functionaltensor(t):
        _assert_wrapped_functional(unwrapped, t)
    return t
