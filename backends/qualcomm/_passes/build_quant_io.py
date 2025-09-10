# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO
from executorch.exir.delegate import executorch_call_delegate

from executorch.exir.pass_base import ExportPass, ProxyValue
from executorch.exir.tensor import TensorSpec
from torch.utils import _pytree as pytree


class BuildQuantIo(ExportPass):
    """
    To make lowering process correct, the pass assign the correct quantized dtype to spec of call_delegate.
    """

    def __init__(self):
        super(BuildQuantIo, self).__init__()

    def _make_spec(self, x):
        if isinstance(x, torch.Tensor):
            return TensorSpec.from_tensor(x)
        elif isinstance(x, (int, bool, float)):
            return x
        else:
            return None

    def placeholder(self, name: str, arg, meta):
        if quantized_dtype := meta.data.get(QCOM_QUANTIZED_IO, None):
            arg = arg.to(dtype=quantized_dtype)
            meta["spec"] = self._make_spec(arg)
        return super().placeholder(name, arg, meta)

    def call_getitem(self, value, key: int, meta):
        meta["spec"] = value.node.meta["spec"][key]
        return super().call_getitem(value, key, meta)

    def call_delegate(self, lowered_module, args, kwargs, meta):
        args_data, _ = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        meta["spec"] = pytree.tree_map(
            self._make_spec,
            executorch_call_delegate(lowered_module, *args_data),
        )
        return super().call_delegate(lowered_module, args, kwargs, meta)
