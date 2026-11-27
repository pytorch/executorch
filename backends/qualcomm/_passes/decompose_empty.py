# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta


def _get_ops(is_edge: bool):
    return (
        {
            "full": exir_ops.edge.aten.full.default,
            "cast": exir_ops.edge.aten._to_copy.default,
        }
        if is_edge
        else {
            "full": torch.ops.aten.full.default,
            "cast": torch.ops.aten.to.dtype,
        }
    )


def _fill_value(dtype: torch.dtype):
    """Return a zero fill scalar and the dtype full.default infers from it.

    full.default has no dtype argument here (see the note in DecomposeEmpty on
    why kwargs are not an option), so its output dtype is derived from the type
    of the python scalar: bool -> bool, int -> int64, float -> the default
    dtype. The caller compares the returned dtype against the dtype it actually
    needs and inserts a cast when they differ.
    """
    if dtype == torch.bool:
        return False, torch.bool
    if dtype.is_floating_point or dtype.is_complex:
        return 0.0, torch.get_default_dtype()
    return 0, torch.int64


class DecomposeEmpty(ExportPass):
    """
    Decompose empty.memory_format / empty_strided into full.default.

    The values are uninitialized, so filling with zero is free. The rewrite is
    applied to every allocation, including a non-contiguous memory_format (e.g.
    channels_last) or an arbitrary stride passed to empty_strided: QNN tensors
    are dense and carry no strides, so the layout cannot survive delegation
    anyway, and nothing in the buffer is readable. Restricting the pass to
    contiguous allocations would instead split the delegate and push the
    remaining empty nodes onto CPU.

    Assumes a plain CPU, non-pinned allocation -- full.default is created
    without device/layout/pin_memory.
    """

    def __init__(self):
        super().__init__()
        self.targets = {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_strided.default,
            exir_ops.edge.aten.empty.memory_format,
            exir_ops.edge.aten.empty_strided.default,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.targets:
                empty_node = node
                val = empty_node.meta["val"]
                shape = list(val.shape)

                is_edge = isinstance(empty_node.target, EdgeOpOverload)
                ops = _get_ops(is_edge)
                fill_value, fill_dtype = _fill_value(val.dtype)

                # The replacement always produces a contiguous tensor, so the
                # propagated FakeTensor must not keep the original strides.
                def cast_meta(meta, dtype=None):
                    out = meta["val"].contiguous()
                    meta["val"] = out if dtype is None else out.to(dtype)
                    return meta

                with graph.inserting_before(empty_node):
                    # Never pass kwargs (dtype/device) to an ATen op here: the
                    # ATen IR requires them to be empty and prepare_pt2e
                    # asserts on it, which this pass would hit since it runs
                    # before the quantizer and full.default is annotated.
                    full_node = graph.create_node(
                        "call_function",
                        ops["full"],
                        (shape, fill_value),
                    )
                    full_node.meta = copy_meta(
                        empty_node.meta,
                        lambda meta, _dtype=fill_dtype: cast_meta(meta, _dtype),
                    )
                    out_node = full_node

                    if fill_dtype != val.dtype:
                        # to.dtype takes its dtype positionally, keeping the
                        # ATen node free of kwargs; the edge dialect uses
                        # _to_copy, where kwargs are expected.
                        cast_args, cast_kwargs = (
                            ((full_node,), {"dtype": val.dtype})
                            if is_edge
                            else ((full_node, val.dtype), {})
                        )
                        cast_node = graph.create_node(
                            "call_function",
                            ops["cast"],
                            cast_args,
                            cast_kwargs,
                        )
                        cast_node.meta = copy_meta(empty_node.meta, cast_meta)
                        out_node = cast_node

                for user in empty_node.users.copy():
                    user.replace_input_with(empty_node, out_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
