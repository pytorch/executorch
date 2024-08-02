# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
from typing import List

import torch

from executorch.exir.schema import (
    AllocationDetails,
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
    SubsegmentOffsets,
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
                            data_buffer_idx=0,
                            allocation_info=AllocationDetails(
                                memory_id=1,
                                memory_offset_high=0,
                                memory_offset_low=16,
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
        constant_buffer=[],
        backend_delegate_data=[],
        segments=[],
        constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
    )


def register_additional_test_aten_ops() -> None:
    # TODO: either mark those ops as canonical in native_functions.yaml,
    # or stop using graphs with those in tests.
    canonical = torch.Tag.core
    torch.ops.aten.max.default.tags.append(canonical)
    torch.ops.aten.sum.default.tags.append(canonical)
    torch.ops.aten.searchsorted.Tensor.tags.append(canonical)
    torch.ops.aten.ones_like.default.tags.append(canonical)
    torch.ops.aten.upsample_nearest2d.default.tags.append(canonical)
    torch.ops.aten.index.Tensor.tags.append(canonical)
    torch.ops.aten.addbmm.default.tags.append(canonical)
