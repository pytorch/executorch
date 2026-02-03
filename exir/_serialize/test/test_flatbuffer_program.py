#!/usr/bin/env fbpython
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

from executorch.exir._serialize._flatbuffer import (
    _program_flatbuffer_to_json,
    _program_json_to_flatbuffer,
)
from executorch.exir._serialize._flatbuffer_program import _program_to_flatbuffer
from executorch.exir._serialize._program import _json_to_program, _program_to_json
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    AllocationDetails,
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Bool,
    BoolList,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    DelegateCall,
    Double,
    DoubleList,
    EValue,
    ExecutionPlan,
    ExtraTensorInfo,
    Frame,
    FrameList,
    FreeCall,
    Instruction,
    Int,
    IntList,
    JumpFalseCall,
    KernelCall,
    MoveCall,
    Null,
    Operator,
    OptionalTensorList,
    Program,
    ScalarType,
    String,
    SubsegmentOffsets,
    Tensor,
    TensorDataLocation,
    TensorList,
    TensorShapeDynamism,
)


class TestFlatbufferProgram(unittest.TestCase):
    def _make_program(self) -> Program:
        return Program(
            version=0,
            execution_plan=[
                ExecutionPlan(
                    name="forward",
                    container_meta_type=ContainerMetadata(
                        encoded_inp_str="encoded_inp_str",
                        encoded_out_str="encoded_out_str",
                    ),
                    values=[
                        EValue(Int(1)),
                        EValue(Bool(True)),
                        EValue(Double(float("inf"))),
                        EValue(String("hello")),
                        EValue(IntList([1, 2, 3])),
                        EValue(DoubleList([1.5, 2.5])),
                        EValue(BoolList([True, False, True])),
                        EValue(TensorList([0, 1])),
                        EValue(OptionalTensorList([-1, 0])),
                        EValue(Null()),
                        EValue(
                            val=Tensor(
                                scalar_type=ScalarType.FLOAT,
                                storage_offset=0,
                                sizes=[2, 2],
                                dim_order=[0, 1],
                                requires_grad=False,
                                layout=0,
                                data_buffer_idx=0,
                                allocation_info=AllocationDetails(
                                    memory_id=1,
                                    memory_offset_high=0,
                                    memory_offset_low=16,
                                ),
                                shape_dynamism=TensorShapeDynamism.STATIC,
                                extra_tensor_info=ExtraTensorInfo(
                                    mutable_data_segments_idx=0,
                                    fully_qualified_name="t0",
                                    location=TensorDataLocation.EXTERNAL,
                                ),
                            )
                        ),
                    ],
                    inputs=[0],
                    outputs=[1],
                    chains=[
                        Chain(
                            inputs=[0],
                            outputs=[1],
                            instructions=[
                                Instruction(KernelCall(op_index=0, args=[0, 1])),
                                Instruction(
                                    DelegateCall(delegate_index=0, args=[1, 2])
                                ),
                                Instruction(MoveCall(move_from=0, move_to=1)),
                                Instruction(
                                    JumpFalseCall(
                                        cond_value_index=1, destination_instruction=0
                                    )
                                ),
                                Instruction(FreeCall(value_index=0)),
                            ],
                            stacktrace=[
                                FrameList(
                                    items=[
                                        Frame(
                                            filename="file.py",
                                            lineno=1,
                                            name="fn",
                                            context="x",
                                        )
                                    ]
                                )
                            ]
                            * 5,
                        )
                    ],
                    operators=[Operator(name="aten::add", overload="Tensor")],
                    delegates=[
                        BackendDelegate(
                            id="delegate0",
                            processed=BackendDelegateDataReference(
                                location=DataLocation.INLINE, index=0
                            ),
                            compile_specs=[CompileSpec(key="k", value=b"v")],
                        ),
                    ],
                    non_const_buffer_sizes=[0, 2**48],
                )
            ],
            constant_buffer=[Buffer(storage=b"abcd")],
            backend_delegate_data=[BackendDelegateInlineData(data=b"delegate-data")],
            segments=[],
            constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
            mutable_data_segments=[],
            named_data=[],
        )

    def _flatbuffer_to_dict(self, flatbuffer_data: bytes) -> dict:
        return json.loads(_program_flatbuffer_to_json(flatbuffer_data))

    def test_roundtrip_via_json(self) -> None:
        program = self._make_program()
        result = _program_to_flatbuffer(
            program, constant_tensor_alignment=32, delegate_alignment=64
        )
        self.assertGreater(len(result.data), 8)
        self.assertEqual(result.data[4:6], b"ET")
        self.assertGreaterEqual(result.max_alignment, 64)

        program2 = _json_to_program(_program_flatbuffer_to_json(result.data))
        self.assertEqual(program2, program)

    def test_flatbuffer_paths_match(self) -> None:
        program = self._make_program()
        cases = [
            (None, None),
            (32, 64),
        ]
        for constant_tensor_alignment, delegate_alignment in cases:
            with self.subTest(
                constant_tensor_alignment=constant_tensor_alignment,
                delegate_alignment=delegate_alignment,
            ):
                result = _program_to_flatbuffer(
                    program,
                    constant_tensor_alignment=constant_tensor_alignment,
                    delegate_alignment=delegate_alignment,
                )
                result2 = _program_json_to_flatbuffer(
                    _program_to_json(program),
                    constant_tensor_alignment=constant_tensor_alignment,
                    delegate_alignment=delegate_alignment,
                )
                direct_dict = self._flatbuffer_to_dict(result.data)
                json_path_dict = self._flatbuffer_to_dict(result2.data)
                self.assertEqual(
                    direct_dict,
                    json_path_dict,
                    "Flatbuffer JSON differs between direct and JSON paths",
                )
                self.assertEqual(result.max_alignment, result2.max_alignment)

    def test_bad_alignment_fails(self) -> None:
        program = Program(
            version=0,
            execution_plan=[],
            constant_buffer=[],
            backend_delegate_data=[],
            segments=[],
            constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        )
        with self.assertRaises(ValueError):
            _program_to_flatbuffer(program, constant_tensor_alignment=3)
