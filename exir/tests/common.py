# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
from typing import Any, Callable, List

import executorch.exir as exir
import torch
import torch.utils._pytree as pytree

from executorch.exir.schema import (
    AllocationDetails,
    Buffer,
    Chain,
    ContainerMetadata,
    EValue,
    ExecutionPlan,
    Instruction,
    Int,
    KernelCall,
    Null,
    Operator,
    Program,
    ScalarType,
    String,
    Tensor,
    TensorShapeDynamism,
)


def get_test_program() -> Program:
    return Program(
        version=0,
        execution_plan=[
            ExecutionPlan(
                name="forward",
                values=[
                    EValue(Int(1)),
                    EValue(Int(0)),
                    EValue(Null()),
                    EValue(String("pass")),
                    EValue(
                        val=Tensor(
                            scalar_type=ScalarType.FLOAT,
                            storage_offset=0,
                            sizes=[2, 2],
                            dim_order=typing.cast(List[bytes], [0, 1]),
                            requires_grad=False,
                            layout=0,
                            constant_buffer_idx=0,
                            allocation_info=AllocationDetails(
                                memory_id=1, memory_offset=16
                            ),
                            shape_dynamism=TensorShapeDynamism.STATIC,
                        )
                    ),
                ],
                inputs=[0],
                outputs=[1],
                chains=[
                    Chain(
                        inputs=[],
                        outputs=[],
                        instructions=[Instruction(KernelCall(op_index=0, args=[0, 1]))],
                        stacktrace=None,
                    )
                ],
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="place", encoded_out_str="place"
                ),
                operators=[Operator(name="aten::add", overload="Tensor")],
                delegates=[],
                non_const_buffer_sizes=[0, 1024],
            )
        ],
        constant_buffer=[Buffer(storage=b"")],
        backend_delegate_data=[],
        segments=[],
    )


# pyre-ignore
def get_graph_module_with_op(op: Callable, args: Any) -> torch.fx.GraphModule:
    """
    Constructs an torch.fx.GraphModule containing just a call to the given op.

    Args:
        op: A callable op
        args: Sample arguments to this given op

    Returns:
        torch.fx.GraphModule with a graph like: inputs -> op -> output
    """

    trace_args, in_spec = pytree.tree_flatten(args)

    graph = torch.fx.Graph()
    with graph.inserting_before(graph._root):
        input_nodes = []
        for i in range(len(trace_args)):
            input_nodes.append(graph.placeholder(f"ph_{i}"))

        op_node = graph.call_function(op, tuple(input_nodes))
        graph.output(op_node)

    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    graph_module.recompile()

    graph_module = exir.capture(graph_module, args).to_edge().module
    return graph_module


def register_additional_test_aten_ops() -> None:
    # TODO: either mark those ops as canonical in native_functions.yaml,
    # or stop using graphs with those in tests.
    canonical = torch.Tag.core  # pyre-ignore
    torch.ops.aten.max.default.tags.append(canonical)
    torch.ops.aten.sum.default.tags.append(canonical)
    torch.ops.aten.searchsorted.Tensor.tags.append(canonical)
    torch.ops.aten.ones_like.default.tags.append(canonical)
    torch.ops.aten.upsample_nearest2d.default.tags.append(canonical)
    torch.ops.aten.index.Tensor.tags.append(canonical)
    torch.ops.aten.addbmm.default.tags.append(canonical)
