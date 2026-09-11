# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
from unittest import mock

import pytest

from executorch.backends.arm.test.tester import test_pipeline
from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    Double,
    EValue,
    ExecutionPlan,
    Instruction,
    Int,
    KernelCall,
    Operator,
    Program,
    ScalarType,
    SubsegmentOffsets,
    Tensor,
    TensorShapeDynamism,
)


def _make_pipeline(tester, pipeline_cls=test_pipeline.BasePipeline):
    pipeline = object.__new__(pipeline_cls)
    pipeline.tester = tester
    pipeline._stages = [
        test_pipeline.PipelineStage(lambda: None, "export"),
        test_pipeline.PipelineStage(lambda: None, "to_executorch"),
    ]
    return pipeline


def _make_pte(delegate_data: bytes) -> bytes:
    return bytes(
        serialize_pte_binary(
            PTEFile(
                program=Program(
                    version=0,
                    execution_plan=[],
                    constant_buffer=[],
                    backend_delegate_data=[
                        BackendDelegateInlineData(data=delegate_data)
                    ],
                    segments=[],
                    constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
                    named_data=[],
                )
            )
        )
    )


def _make_delegated_pte(
    backend_payloads: list[bytes],
    delegate_refs: list[tuple[str, int]],
) -> bytes:
    return bytes(
        serialize_pte_binary(
            PTEFile(
                program=Program(
                    version=0,
                    execution_plan=[
                        ExecutionPlan(
                            name="forward",
                            values=[],
                            inputs=[],
                            outputs=[],
                            chains=[],
                            container_meta_type=ContainerMetadata(
                                encoded_inp_str="", encoded_out_str=""
                            ),
                            operators=[],
                            delegates=[
                                BackendDelegate(
                                    id=delegate_id,
                                    processed=BackendDelegateDataReference(
                                        location=DataLocation.INLINE,
                                        index=payload_index,
                                    ),
                                    compile_specs=[],
                                )
                                for delegate_id, payload_index in delegate_refs
                            ],
                            non_const_buffer_sizes=[],
                        )
                    ],
                    constant_buffer=[],
                    backend_delegate_data=[
                        BackendDelegateInlineData(data=payload)
                        for payload in backend_payloads
                    ],
                    segments=[],
                    constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
                    named_data=[],
                )
            )
        )
    )


def _make_delegated_pte_with_values(
    backend_payload: bytes,
    values: list[EValue],
) -> bytes:
    return bytes(
        serialize_pte_binary(
            PTEFile(
                program=Program(
                    version=0,
                    execution_plan=[
                        ExecutionPlan(
                            name="forward",
                            values=values,
                            inputs=[],
                            outputs=[],
                            chains=[],
                            container_meta_type=ContainerMetadata(
                                encoded_inp_str="", encoded_out_str=""
                            ),
                            operators=[],
                            delegates=[
                                BackendDelegate(
                                    id="TOSABackend",
                                    processed=BackendDelegateDataReference(
                                        location=DataLocation.INLINE,
                                        index=0,
                                    ),
                                    compile_specs=[],
                                )
                            ],
                            non_const_buffer_sizes=[],
                        )
                    ],
                    constant_buffer=[],
                    backend_delegate_data=[
                        BackendDelegateInlineData(data=backend_payload)
                    ],
                    segments=[],
                    constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
                    named_data=[],
                )
            )
        )
    )


def _make_tensor_pte(values: list[float], sizes: list[int] | None = None) -> bytes:
    if sizes is None:
        sizes = [len(values)]
    weights = struct.pack(f"<{len(values)}f", *values)
    return bytes(
        serialize_pte_binary(
            PTEFile(
                program=Program(
                    version=0,
                    execution_plan=[
                        ExecutionPlan(
                            name="forward",
                            values=[
                                EValue(
                                    val=Tensor(
                                        scalar_type=ScalarType.FLOAT,
                                        storage_offset=0,
                                        sizes=sizes,
                                        dim_order=list(range(len(sizes))),
                                        requires_grad=False,
                                        layout=0,
                                        data_buffer_idx=1,
                                        allocation_info=None,
                                        shape_dynamism=TensorShapeDynamism.STATIC,
                                    )
                                ),
                                EValue(Int(0)),
                            ],
                            inputs=[1],
                            outputs=[0],
                            chains=[
                                Chain(
                                    inputs=[],
                                    outputs=[],
                                    instructions=[
                                        Instruction(KernelCall(op_index=0, args=[0, 1]))
                                    ],
                                    stacktrace=None,
                                )
                            ],
                            container_meta_type=ContainerMetadata(
                                encoded_inp_str="", encoded_out_str=""
                            ),
                            operators=[Operator(name="aten::add", overload="out")],
                            delegates=[],
                            non_const_buffer_sizes=[0, 64],
                        )
                    ],
                    constant_buffer=[Buffer(storage=b""), Buffer(storage=weights)],
                    backend_delegate_data=[],
                    segments=[],
                    constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
                    named_data=[],
                )
            )
        )
    )


def _make_double_pte(value: float, op_name: str = "aten::add") -> bytes:
    return bytes(
        serialize_pte_binary(
            PTEFile(
                program=Program(
                    version=0,
                    execution_plan=[
                        ExecutionPlan(
                            name="forward",
                            values=[EValue(Double(value))],
                            inputs=[],
                            outputs=[],
                            chains=[],
                            container_meta_type=ContainerMetadata(
                                encoded_inp_str="", encoded_out_str=""
                            ),
                            operators=[Operator(name=op_name, overload="out")],
                            delegates=[],
                            non_const_buffer_sizes=[],
                        )
                    ],
                    constant_buffer=[],
                    backend_delegate_data=[],
                    segments=[],
                    constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
                    named_data=[],
                )
            )
        )
    )


def test_diff_pte_adds_stage_after_to_executorch():
    tester = mock.Mock()
    tester.get_artifact.side_effect = AssertionError(
        "diff_pte read the to_executorch artifact too early"
    )
    pipeline = _make_pipeline(tester)

    assert pipeline.diff_pte(b"reference") is pipeline

    assert [stage.id for stage in pipeline._stages] == [
        "export",
        "to_executorch",
        "_compare_pte.diff_pte",
    ]
    tester.get_artifact.assert_not_called()


def test_diff_pte_stage_compares_to_executorch_artifact():
    pte = _make_pte(b"delegate")
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=pte)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(pte)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()

    tester.get_artifact.assert_called_once_with(test_pipeline.StageType.TO_EXECUTORCH)


def test_base_pipeline_has_no_backend_specific_pte_diagnostics():
    pipeline = object.__new__(test_pipeline.BasePipeline)

    assert (
        pipeline._format_extra_pte_diff(
            _make_pte(b"delegate-a"),
            _make_pte(b"delegate-b"),
            max_samples=10,
        )
        == []
    )


def test_diff_pte_fails_on_tensor_drift_by_default():
    reference = _make_tensor_pte([1.0, 2.0, 3.0, 4.0])
    candidate = _make_tensor_pte([1.0, 2.0000005, 3.0, 4.0])
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_allows_tensor_drift_with_atol():
    reference = _make_tensor_pte([1.0, 2.0, 3.0, 4.0])
    candidate = _make_tensor_pte([1.0, 2.0000005, 3.0, 4.0])
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_allows_tensor_drift_with_rtol():
    reference = _make_tensor_pte([1.0, 100.0, 3.0, 4.0])
    candidate = _make_tensor_pte([1.0, 100.00005, 3.0, 4.0])
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, rtol=1e-6)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_when_tensor_drift_exceeds_tolerance():
    reference = _make_tensor_pte([1.0, 2.0, 3.0, 4.0])
    candidate = _make_tensor_pte([1.0, 2.0001, 3.0, 4.0])
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_on_tensor_metadata_diff_even_with_tolerance():
    reference = _make_tensor_pte([1.0, 2.0, 3.0, 4.0], sizes=[2, 2])
    candidate = _make_tensor_pte([1.0, 2.0000005, 3.0, 4.0], sizes=[4])
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_on_double_drift_by_default():
    reference = _make_double_pte(0.1)
    candidate = _make_double_pte(0.1000001)
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_allows_double_drift_with_atol():
    reference = _make_double_pte(0.1)
    candidate = _make_double_pte(0.1000001)
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_on_delegate_payload_drift_even_with_tolerance():
    reference = _make_delegated_pte_with_values(
        b"delegate-a",
        [EValue(Double(1.0))],
    )
    candidate = _make_delegated_pte_with_values(
        b"delegate-b",
        [EValue(Double(1.0000001))],
    )
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_allows_double_drift_with_rtol():
    reference = _make_double_pte(1.0)
    candidate = _make_double_pte(1.0000001)
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, rtol=1e-6)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_when_double_drift_exceeds_tolerance():
    reference = _make_double_pte(0.1)
    candidate = _make_double_pte(0.1001)
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_diff_pte_fails_on_structural_diff_even_with_tolerance():
    reference = _make_double_pte(0.1, op_name="aten::add")
    candidate = _make_double_pte(0.1000001, op_name="aten::mul")
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester)

    pipeline.diff_pte(candidate, atol=1e-6)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_tosa_delegate_diff_prints_operator_type_mismatch():
    with mock.patch.object(
        test_pipeline,
        "_read_tosa_ops",
        side_effect=[
            ["ADD", "CONV2D"],
            ["ADD", "CONST"],
        ],
    ):
        diff = test_pipeline.TOSAPipeline._format_extra_pte_diff(
            object.__new__(test_pipeline.TOSAPipeline),
            _make_delegated_pte(
                [b"unused-a", b"other-a", b"delegate-a"],
                [("OtherBackend", 1), ("TOSABackend", 2)],
            ),
            _make_delegated_pte(
                [b"unused-b", b"other-b", b"delegate-b"],
                [("OtherBackend", 1), ("TOSABackend", 2)],
            ),
            max_samples=10,
        )

    assert "Backend delegate TOSA op differences:" in diff
    assert "  [0]: 10 B vs 10 B" in diff
    assert "    TOSA op count: 2 vs 2" in diff
    assert "    TOSA ops only in A: CONV2D" in diff
    assert "    TOSA ops only in B: CONST" in diff
    assert "    TOSA operator type mismatch: op[1]: CONV2D vs CONST" in diff


def test_tosa_delegate_diff_ignores_non_tosa_delegate_payloads():
    diff = test_pipeline.TOSAPipeline._format_extra_pte_diff(
        object.__new__(test_pipeline.TOSAPipeline),
        _make_delegated_pte(
            [b"other-a"],
            [("OtherBackend", 0)],
        ),
        _make_delegated_pte(
            [b"other-b"],
            [("OtherBackend", 0)],
        ),
        max_samples=10,
    )

    assert diff == []


def test_tosa_diff_pte_allows_identical_tosa_delegate_payloads():
    reference = _make_delegated_pte_with_values(
        b"delegate",
        [EValue(Int(1))],
    )
    candidate = _make_delegated_pte_with_values(
        b"delegate",
        [EValue(Int(1)), EValue(Int(2))],
    )
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester, test_pipeline.TOSAPipeline)

    pipeline.diff_pte(candidate)
    pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_tosa_diff_pte_fails_when_no_tosa_delegate_payloads_match():
    reference = _make_delegated_pte(
        [b"other"],
        [("OtherBackend", 0)],
    )
    candidate = _make_delegated_pte(
        [b"other"],
        [("OtherBackend", 0)],
    )
    candidate = candidate + b"\0"
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester, test_pipeline.TOSAPipeline)

    pipeline.diff_pte(candidate)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_tosa_diff_pte_allows_changed_payloads_with_identical_tosa_ops():
    reference = _make_delegated_pte_with_values(
        b"delegate-a",
        [EValue(Int(1))],
    )
    candidate = _make_delegated_pte_with_values(
        b"delegate-b",
        [EValue(Int(1)), EValue(Int(2))],
    )
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester, test_pipeline.TOSAPipeline)

    with mock.patch.object(
        test_pipeline,
        "_read_tosa_ops",
        side_effect=[
            ["ADD", "CONV2D"],
            ["ADD", "CONV2D"],
        ],
    ):
        pipeline.diff_pte(candidate)
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_tosa_diff_pte_fails_on_changed_tosa_ops():
    reference = _make_delegated_pte_with_values(
        b"delegate-a",
        [EValue(Int(1))],
    )
    candidate = _make_delegated_pte_with_values(
        b"delegate-b",
        [EValue(Int(1)), EValue(Int(2))],
    )
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester, test_pipeline.TOSAPipeline)

    with mock.patch.object(
        test_pipeline,
        "_read_tosa_ops",
        side_effect=[
            ["ADD", "CONV2D"],
            ["ADD", "CONST"],
            ["ADD", "CONV2D"],
            ["ADD", "CONST"],
        ],
    ):
        pipeline.diff_pte(candidate)
        with pytest.raises(AssertionError):
            pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()


def test_tosa_diff_pte_fails_on_changed_tosa_delegate_payloads():
    reference = _make_delegated_pte_with_values(
        b"delegate-a",
        [EValue(Int(1))],
    )
    candidate = _make_delegated_pte_with_values(
        b"delegate-b",
        [EValue(Int(1))],
    )
    tester = mock.Mock()
    tester.get_artifact.return_value = mock.Mock(buffer=reference)
    pipeline = _make_pipeline(tester, test_pipeline.TOSAPipeline)

    pipeline.diff_pte(candidate)
    with pytest.raises(AssertionError):
        pipeline._stages[pipeline.find_pos("_compare_pte.diff_pte")]()
