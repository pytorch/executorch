# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from pathlib import Path
from typing import Any

from executorch.exir._serialize._serialize import serialize_for_executorch
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    AllocationDetails,
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Chain,
    ContainerMetadata,
    DataLocation,
    DelegateCall,
    EValue,
    ExecutionPlan,
    Instruction,
    String,
    Tensor,
    TensorShapeDynamism,
)
from executorch.exir.scalar_type import ScalarType


TOKENIZER_BACKEND_ID = "TokenizerBackend"
TOKENIZER_METHOD_NAME = "tokenize"


def _allocation_info(memory_id: int, memory_offset: int) -> AllocationDetails:
    return AllocationDetails(
        memory_id=memory_id,
        memory_offset_low=memory_offset & ((1 << 32) - 1),
        memory_offset_high=memory_offset >> 32,
    )


def _make_token_tensor(max_context_length: int) -> Tensor:
    if max_context_length <= 0:
        raise ValueError(
            f"max_context_length must be positive, got {max_context_length}"
        )
    return Tensor(
        scalar_type=ScalarType.LONG,
        storage_offset=0,
        sizes=[max_context_length],
        dim_order=[0],
        requires_grad=False,
        layout=0,
        data_buffer_idx=0,
        allocation_info=_allocation_info(memory_id=1, memory_offset=0),
        shape_dynamism=TensorShapeDynamism.DYNAMIC_BOUND,
    )


def append_tokenizer_delegate_method(
    executorch_program_manager: Any,
    tokenizer_path: str,
    max_context_length: int,
    method_name: str = TOKENIZER_METHOD_NAME,
    bos: int = 0,
    eos: int = 0,
) -> None:
    """
    Add a tokenizer entry point directly to an ExecuTorch program.

    The method takes one string EValue and returns one int64 token tensor. The
    tensor is memory planned to the model's max context length and resized by
    the runtime tokenizer delegate to the actual token count.
    """
    tokenizer_bytes = Path(tokenizer_path).read_bytes()
    program = executorch_program_manager._emitter_output.program

    if any(plan.name == method_name for plan in program.execution_plan):
        raise ValueError(f"Program already has a method named {method_name}")

    delegate_data_index = len(program.backend_delegate_data)
    program.backend_delegate_data.append(
        BackendDelegateInlineData(data=tokenizer_bytes)
    )

    delegate = BackendDelegate(
        id=TOKENIZER_BACKEND_ID,
        processed=BackendDelegateDataReference(
            location=DataLocation.INLINE,
            index=delegate_data_index,
        ),
        compile_specs=[
            CompileSpec("max_context_length", str(max_context_length).encode()),
            CompileSpec("bos", str(bos).encode()),
            CompileSpec("eos", str(eos).encode()),
        ],
    )

    input_id = 0
    output_id = 1
    plan = ExecutionPlan(
        name=method_name,
        values=[
            EValue(String("")),
            EValue(_make_token_tensor(max_context_length)),
        ],
        inputs=[input_id],
        outputs=[output_id],
        chains=[
            Chain(
                inputs=[input_id],
                outputs=[output_id],
                instructions=[
                    Instruction(
                        DelegateCall(
                            delegate_index=0,
                            args=[input_id, output_id],
                        )
                    )
                ],
                stacktrace=None,
            )
        ],
        operators=[],
        delegates=[delegate],
        non_const_buffer_sizes=[0, max_context_length * 8],
        container_meta_type=ContainerMetadata("", ""),
    )
    program.execution_plan.append(plan)

    executorch_program_manager._pte_data, executorch_program_manager._tensor_data = (
        serialize_for_executorch(
            executorch_program_manager._emitter_output,
            executorch_program_manager._backend_config,
            executorch_program_manager._data_serializer,
            executorch_program_manager._named_data,
        )
    )
    executorch_program_manager._buffer = None
