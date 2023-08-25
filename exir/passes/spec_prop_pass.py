# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from executorch.exir.tensor import TensorSpec
from torch.utils import _pytree as pytree


# pyre-ignore
def make_spec(x):
    if isinstance(x, torch.Tensor):
        return TensorSpec.from_tensor(x)
    elif isinstance(x, (int, bool, float)):
        return x
    else:
        return None


class SpecPropPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def on_attr(self, attr: ProxyValue) -> None:
        attr.node.meta["spec"] = pytree.tree_map_only(
            torch.Tensor,
            make_spec,
            attr.data,
        )

    # pyre-ignore
    def placeholder(self, name: str, arg, meta):
        meta["spec"] = make_spec(arg)
        return super().placeholder(name, arg, meta)

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        meta["spec"] = pytree.tree_map(make_spec, op(*args_data, **kwargs_data))
        return super().call_operator(op, args, kwargs, meta)

    # pyre-ignore
    def call_getitem(self, value, key: int, meta):
        meta["spec"] = value.node.meta["spec"][key]
        return super().call_getitem(value, key, meta)

    # pyre-ignore
    def call_cond(self, pred, true_fn, false_fn, inputs, meta):
        # true_fn/false_fn return tensors of the same shape, so we can pick
        # either one here.
        *_, true_out_node = true_fn.graph.nodes
        meta["spec"] = pytree.tree_map(make_spec, true_out_node.meta["val"])
        return super().call_cond(pred, true_fn, false_fn, inputs, meta)

    def call_map(
        self,
        f: torch.fx.GraphModule,
        num_args: int,
        args: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, args)
        xs_data = args_data[:num_args]
        *_, body_out_node = f.graph.nodes
        body_out_node_fake_tensor = body_out_node.meta["val"]
        map_fake_tensor = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.new_empty(xs_data[0].size(0), *x.shape),
            body_out_node_fake_tensor,
        )
        meta["spec"] = pytree.tree_map(make_spec, map_fake_tensor)
        return super().call_map(f, num_args, args, meta)

    # pyre-ignore
    def call_delegate(self, lowered_module, args, kwargs, meta):
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        # If spec is missing, re-genenrate it with args data
        if "spec" not in meta:
            meta["spec"] = pytree.tree_map(
                make_spec,
                executorch_call_delegate(lowered_module, *args_data),
            )
        return super().call_delegate(lowered_module, args, kwargs, meta)

    # pyre-ignore
    def output(self, results, meta):
        # pyre-ignore
        def get_spec(x):
            if isinstance(x, ProxyValue):
                return x.node.meta["spec"]
            else:
                return make_spec(x)

        meta["spec"] = pytree.tree_map(get_spec, results)
        return super().output(results, meta)
