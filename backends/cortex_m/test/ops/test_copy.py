# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class CortexMCopy(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_copy_default": 1,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_copy_default": 0,
        "executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default": 1,
    }

    def forward(self, dst, src):
        return torch.ops.aten.copy.default(dst, src)


class CortexMConstantCopy(CortexMCopy):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "constant", torch.arange(6, dtype=torch.int8).reshape(3, 2).T
        )

    def forward(self, dst):
        return torch.ops.aten.copy.default(dst, self.constant)


_values = torch.arange(24, dtype=torch.int8)
_nchw = _values.reshape(1, 3, 2, 4)
_nhwc = _nchw.to(memory_format=torch.channels_last)
_singleton = _values[:6].reshape(1, 1, 2, 3)

test_cases = {
    "contiguous": McuTestCase(CortexMCopy(), (torch.full_like(_nchw, -1), _nchw)),
    "channels_last": McuTestCase(CortexMCopy(), (torch.full_like(_nhwc, -1), _nhwc)),
    "contiguous_to_channels_last": McuTestCase(
        CortexMCopy(), (torch.full_like(_nhwc, -1), _nchw)
    ),
    "channels_last_to_contiguous": McuTestCase(
        CortexMCopy(), (torch.full_like(_nchw, -1), _nhwc)
    ),
    "singleton_channels_last": McuTestCase(
        CortexMCopy(),
        (
            torch.full_like(_singleton, -1),
            _singleton.to(memory_format=torch.channels_last),
        ),
    ),
    "scalar": McuTestCase(CortexMCopy(), (torch.full_like(_values[5], -1), _values[5])),
    "rank1": McuTestCase(CortexMCopy(), (torch.full_like(_values, -1), _values)),
    "rank2": McuTestCase(
        CortexMCopy(),
        (torch.full((4, 6), -1, dtype=torch.int8), _values.reshape(4, 6)),
    ),
    "rank3": McuTestCase(
        CortexMCopy(),
        (
            torch.full((2, 3, 4), -1, dtype=torch.int8),
            _values.reshape(2, 3, 4),
        ),
    ),
    "transposed_constant": McuTestCase(
        CortexMConstantCopy(), (torch.zeros(2, 3, dtype=torch.int8),)
    ),
}

fallback_test_cases = {
    "float": McuTestCase(CortexMCopy(), (torch.zeros(2, 3), torch.ones(2, 3))),
    "int64": McuTestCase(
        CortexMCopy(),
        (torch.zeros(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.int64)),
    ),
    "conversion": McuTestCase(
        CortexMCopy(), (torch.zeros(2, 3, dtype=torch.int8), torch.ones(2, 3))
    ),
    "broadcast": McuTestCase(
        CortexMCopy(),
        (torch.zeros(2, 3, dtype=torch.int8), torch.ones(1, 3, dtype=torch.int8)),
    ),
    "rank5": McuTestCase(
        CortexMCopy(),
        (
            torch.zeros(1, 2, 3, 4, 5, dtype=torch.int8),
            torch.ones(1, 2, 3, 4, 5, dtype=torch.int8),
        ),
    ),
    "large_singleton_stride": McuTestCase(
        CortexMCopy(),
        (
            torch.zeros(1, 3, dtype=torch.int8),
            torch.empty_strided((1, 3), (2**31, 1), dtype=torch.int8).fill_(4),
        ),
    ),
}


@parametrize("test_case", test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_copy(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        atol=0,
        use_explicit_layout=use_explicit_layout,
    )
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer


@parametrize("test_case", test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_implementation_copy(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(atol=0, use_explicit_layout=use_explicit_layout)


@parametrize("test_case", fallback_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_copy_fallback(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        {
            **test_case.model.ops_before_transforms,
            "executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default": 0,
        },
        atol=0,
        use_explicit_layout=use_explicit_layout,
    )
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer


@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_copy_dynamic_fallback(use_explicit_layout):
    batch = torch.export.Dim("batch", min=2, max=5)
    inputs = (torch.zeros(3, 4, dtype=torch.int8), torch.ones(3, 4, dtype=torch.int8))
    exported = torch.export.export(
        CortexMCopy(), inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    edge = to_edge(exported, compile_config=cortex_m_edge_compile_config()).transform(
        CortexMPassManager(use_explicit_layout=use_explicit_layout)
    )
    assert not any(
        node.target == exir_ops.edge.cortex_m.strided_copy.default
        for node in edge.exported_program().graph.nodes
    )
    for size in (2, 4, 5):
        args = (
            torch.zeros(size, 4, dtype=torch.int8),
            torch.ones(size, 4, dtype=torch.int8),
        )
        torch.testing.assert_close(
            edge.exported_program().module()(*args), CortexMCopy()(*args)
        )
    assert edge.to_executorch().buffer


class CortexMCopyPlan(torch.nn.Module):
    def __init__(
        self,
        expected,
        shapes,
        src_strides,
        src_offsets,
        dst_origins,
        dst_steps,
        dst_strides=None,
    ):
        super().__init__()
        self.expected = expected.contiguous()
        self.shapes = shapes
        self.src_strides = src_strides
        self.dst_strides = (
            list(self.expected.stride()) if dst_strides is None else dst_strides
        )
        self.src_offsets = src_offsets
        self.dst_origins = dst_origins
        self.dst_steps = dst_steps

    def forward(self, *inputs):
        return torch.ops.cortex_m.strided_copy.default(
            list(inputs),
            self.shapes,
            self.src_strides,
            self.dst_strides,
            self.src_offsets,
            self.dst_origins,
            self.dst_steps,
        )


_plan_matrix = torch.arange(6, dtype=torch.int8).reshape(2, 3)
_plan_image = torch.arange(48, dtype=torch.int8).reshape(1, 8, 2, 3)
_plan_scatter = torch.arange(22, dtype=torch.int8).reshape(2, 11)
_plan_rank16 = torch.randint(
    -128, 128, [2] * 16, dtype=torch.int8, generator=torch.Generator().manual_seed(0)
)
_phase_origins = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]

# These plans exercise the runtime contract without adding more ATen lowerings.
copy_plan_test_cases = {
    "scalar_offset": McuTestCase(
        CortexMCopyPlan(
            _values[5],
            shapes=[],
            src_strides=[],
            src_offsets=[5],
            dst_origins=[],
            dst_steps=[],
        ),
        (_values,),
    ),
    "singleton_extreme_fields": McuTestCase(
        CortexMCopyPlan(
            _values[5].reshape(1, 1),
            shapes=[1, 1],
            src_strides=[-(2**31), 2**31 - 1],
            src_offsets=[5],
            dst_origins=[0, 0],
            dst_steps=[2**31 - 1] * 2,
            dst_strides=[2**31 - 1] * 2,
        ),
        (_values,),
    ),
    "singleton_zero_stride": McuTestCase(
        CortexMCopyPlan(
            _values[:3].reshape(1, 3),
            shapes=[1, 3],
            src_strides=[3, 1],
            src_offsets=[0],
            dst_origins=[0, 0],
            dst_steps=[1, 1],
            dst_strides=[0, 1],
        ),
        (_values[:3].reshape(1, 3),),
    ),
    "flip": McuTestCase(
        CortexMCopyPlan(
            _plan_matrix.flip(1),
            shapes=[2, 3],
            src_strides=[3, -1],
            src_offsets=[2],
            dst_origins=[0, 0],
            dst_steps=[1, 1],
        ),
        (_plan_matrix,),
    ),
    "rot90": McuTestCase(
        CortexMCopyPlan(
            _plan_matrix.rot90(1, (0, 1)),
            shapes=[3, 2],
            src_strides=[-1, 3],
            src_offsets=[2],
            dst_origins=[0, 0],
            dst_steps=[1, 1],
        ),
        (_plan_matrix,),
    ),
    "expand": McuTestCase(
        CortexMCopyPlan(
            _values[5].expand(2, 3),
            shapes=[2, 3],
            src_strides=[0, 0],
            src_offsets=[0],
            dst_origins=[0, 0],
            dst_steps=[1, 1],
        ),
        (_values[5],),
    ),
    "pixel_shuffle": McuTestCase(
        CortexMCopyPlan(
            torch.pixel_shuffle(_plan_image, 2),
            shapes=[1, 2, 2, 3] * 4,
            src_strides=[48, 24, 3, 1] * 4,
            src_offsets=[0, 6, 12, 18],
            dst_origins=_phase_origins,
            dst_steps=[1, 1, 2, 2],
        ),
        (_plan_image,) * 4,
    ),
    "pixel_unshuffle": McuTestCase(
        CortexMCopyPlan(
            torch.pixel_unshuffle(_plan_image.reshape(1, 2, 4, 6), 2),
            shapes=[1, 2, 2, 3] * 4,
            src_strides=[48, 24, 12, 2] * 4,
            src_offsets=[0, 1, 6, 7],
            dst_origins=[0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0],
            dst_steps=[1, 4, 1, 1],
        ),
        (_plan_image.reshape(1, 2, 4, 6),) * 4,
    ),
    "slice_scatter_ragged": McuTestCase(
        CortexMCopyPlan(
            torch.slice_scatter(_plan_scatter, _plan_matrix, 1, 1, 10, 3),
            shapes=[2, 4, 2, 3, 2, 3, 2, 1],
            src_strides=[11, 3, 3, 1, 11, 3, 11, 3],
            src_offsets=[0, 0, 2, 10],
            dst_origins=[0, 0, 0, 1, 0, 2, 0, 10],
            dst_steps=[1, 3],
        ),
        (_plan_scatter, _plan_matrix, _plan_scatter, _plan_scatter),
    ),
    "unfold_rank5": McuTestCase(
        CortexMCopyPlan(
            _nchw.unfold(3, 3, 1),
            shapes=[1, 3, 2, 2, 3],
            src_strides=[24, 8, 4, 1, 1],
            src_offsets=[0],
            dst_origins=[0] * 5,
            dst_steps=[1] * 5,
        ),
        (_nchw,),
    ),
    "rank16": McuTestCase(
        CortexMCopyPlan(
            _plan_rank16.flip(list(range(16))),
            shapes=[2] * 16,
            src_strides=[-stride for stride in _plan_rank16.stride()],
            src_offsets=[_plan_rank16.numel() - 1],
            dst_origins=[0] * 16,
            dst_steps=[1] * 16,
        ),
        (_plan_rank16,),
    ),
}


@parametrize("test_case", copy_plan_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_copy_plan(test_case, use_explicit_layout, cortex_m_target):
    torch.testing.assert_close(
        test_case.model(*test_case.example_inputs),
        test_case.model.expected,
        rtol=0,
        atol=0,
    )
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    ops = {"executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default": 1}
    tester.test_dialect(ops, ops, atol=0, use_explicit_layout=use_explicit_layout)
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer


@parametrize("test_case", copy_plan_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_implementation_copy_plan(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(atol=0, use_explicit_layout=use_explicit_layout)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("destination_stride", [0, 2**31 - 1])
def test_copy_plan_empty(strict, destination_stride):
    model = CortexMCopyPlan(
        _values[:0],
        shapes=[0],
        src_strides=[-(2**31)],
        src_offsets=[2**31 - 1],
        dst_origins=[0],
        dst_steps=[2**31 - 1],
        dst_strides=[destination_stride],
    )
    inputs = (_values[:0],)
    torch.testing.assert_close(model(*inputs), model.expected)
    exported = torch.export.export(model, inputs, strict=strict)
    edge = to_edge(exported, compile_config=cortex_m_edge_compile_config())
    assert edge.to_executorch().buffer


invalid_copy_plan_test_cases = {
    "no_inputs": ({"inputs": ()}, "invalid copy plan"),
    "missing_shapes": ({"shapes": []}, "invalid copy plan"),
    "missing_source_strides": ({"src_strides": []}, "invalid copy plan"),
    "missing_source_offsets": ({"src_offsets": []}, "invalid copy plan"),
    "missing_destination_origins": ({"dst_origins": []}, "invalid copy plan"),
    "missing_destination_steps": ({"dst_steps": []}, "invalid copy plan"),
    "rank17": (
        {
            "shapes": [1] * 17,
            "src_strides": [0] * 17,
            "dst_strides": [1] * 17,
            "dst_origins": [0] * 17,
            "dst_steps": [1] * 17,
        },
        "invalid copy plan",
    ),
    "negative_count": ({"shapes": [-1]}, "int32 ranges"),
    "large_count": ({"shapes": [2**31]}, "int32 ranges"),
    "large_source_stride": ({"src_strides": [2**31]}, "int32 ranges"),
    "small_source_stride": ({"src_strides": [-(2**31) - 1]}, "int32 ranges"),
    "negative_source_offset": ({"src_offsets": [-1]}, "int32 ranges"),
    "large_source_offset": ({"src_offsets": [2**31]}, "int32 ranges"),
    "negative_destination_stride": ({"dst_strides": [-1]}, "int32 ranges"),
    "large_destination_stride": ({"dst_strides": [2**31]}, "int32 ranges"),
    "negative_destination_origin": ({"dst_origins": [-1]}, "int32 ranges"),
    "large_destination_origin": ({"dst_origins": [2**31]}, "int32 ranges"),
    "zero_destination_step": ({"dst_steps": [0]}, "int32 ranges"),
    "negative_destination_step": ({"dst_steps": [-1]}, "int32 ranges"),
    "large_destination_step": ({"dst_steps": [2**31]}, "int32 ranges"),
    "float": ({"inputs": (torch.ones(4),)}, "dense int8"),
    "non_dense_input": ({"inputs": (_values[:8:2],)}, "dense int8"),
    "transposed_input": (
        {"inputs": (_values[:4].reshape(2, 2).T,)},
        "source layout",
    ),
    "channels_last_3d_input": (
        {
            "inputs": (
                _values.reshape(1, 2, 2, 2, 3).to(memory_format=torch.channels_last_3d),
            )
        },
        "source layout",
    ),
    "input_rank17": (
        {"inputs": (torch.zeros([1] * 17, dtype=torch.int8),)},
        "Cortex-M tensor limits",
    ),
    "input_numel_overflow": (
        {"inputs": (torch.empty(65536, 32768, dtype=torch.int8, device="meta"),)},
        "Cortex-M tensor limits",
    ),
    "output_extent_overflow": ({"dst_origins": [2**31 - 1]}, "output shape"),
    "output_numel_overflow": (
        {
            "shapes": [65536, 32768],
            "src_strides": [0, 0],
            "dst_strides": [32768, 1],
            "dst_origins": [0, 0],
            "dst_steps": [1, 1],
        },
        "output shape",
    ),
    "non_dense_output": ({"dst_strides": [2]}, "destination must be dense"),
    "aliased_output": ({"dst_strides": [0]}, "destination must be dense"),
    "source_offset_oob": ({"src_offsets": [4]}, "source exceeds tensor bounds"),
    "source_stride_oob": ({"src_strides": [2]}, "source exceeds tensor bounds"),
    "source_underflow": ({"src_strides": [-1]}, "source exceeds tensor bounds"),
    "coverage_hole": ({"shapes": [2], "dst_origins": [1]}, "cover the output"),
    "overlap_with_hole": (
        {
            "inputs": (_values[:4],) * 3,
            "shapes": [2, 1, 1],
            "src_strides": [1, 1, 1],
            "src_offsets": [0, 0, 0],
            "dst_origins": [0, 1, 3],
        },
        "regions overlap",
    ),
}


@parametrize("test_case", invalid_copy_plan_test_cases)
@pytest.mark.parametrize("strict", [False, True])
def test_copy_plan_rejected_during_export(test_case, strict):
    overrides, message = test_case
    plan = {
        "shapes": [4],
        "src_strides": [1],
        "src_offsets": [0],
        "dst_origins": [0],
        "dst_steps": [1],
        **overrides,
    }
    inputs = plan.pop("inputs", (_values[:4],))
    model = CortexMCopyPlan(_values[:4], **plan)
    with pytest.raises(ValueError, match=f"strided_copy: .*{message}"):
        model(*inputs)
    with pytest.raises((ValueError, RuntimeError), match=f"strided_copy: .*{message}"):
        torch.export.export(model, inputs, strict=strict)


@pytest.mark.parametrize("strict", [False, True])
def test_copy_plan_requires_static_input(strict):
    model = CortexMCopyPlan(
        _values[:4],
        shapes=[4],
        src_strides=[1],
        src_offsets=[0],
        dst_origins=[0],
        dst_steps=[1],
    )
    with pytest.raises((ValueError, RuntimeError), match="strided_copy: .*static"):
        torch.export.export(
            model,
            (_values[:4],),
            dynamic_shapes=(({0: torch.export.Dim("length", min=4, max=8)},),),
            strict=strict,
        )


@pytest.mark.parametrize("strict", [False, True])
def test_copy_plan_rejects_transposed_constant(strict):
    class ConstantCopyPlan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("constant", _values[:6].reshape(3, 2).T)

        def forward(self, x):
            return torch.ops.cortex_m.strided_copy.default(
                [x, self.constant],
                [2, 3, 2, 3],
                [3, 1, 1, 2],
                [3, 1],
                [0, 0],
                [0, 0, 2, 0],
                [1, 1],
            )

    with pytest.raises(
        (ValueError, RuntimeError), match="strided_copy: .*source layout"
    ):
        torch.export.export(
            ConstantCopyPlan(), (torch.zeros(2, 3, dtype=torch.int8),), strict=strict
        )


@pytest.mark.parametrize(
    "arg_index, value, message",
    [(4, [4], "source exceeds tensor bounds"), (3, [2], "destination must be dense")],
)
def test_copy_plan_revalidated_before_emission(arg_index, value, message):
    inputs = (_values[:4], _values[:4])
    exported = torch.export.export(CortexMCopy(), inputs)
    edge = to_edge(exported, compile_config=cortex_m_edge_compile_config()).transform(
        CortexMPassManager()
    )
    [copy_node] = [
        node
        for node in edge.exported_program().graph.nodes
        if node.target == exir_ops.edge.cortex_m.strided_copy.default
    ]
    args = list(copy_node.args)
    args[arg_index] = value
    copy_node.args = tuple(args)
    edge.exported_program().graph_module.recompile()
    with pytest.raises((ValueError, RuntimeError), match=f"strided_copy: .*{message}"):
        edge.to_executorch()


@pytest.mark.parametrize(
    "origins, message",
    [
        ([0, 2, 4], "regions must cover the output"),
        ([0, 1, 3], "destination regions overlap"),
    ],
)
def test_copy_plan_regions_revalidated_before_emission(origins, message):
    model = CortexMCopyPlan(
        _values[:4],
        shapes=[2, 1, 1],
        src_strides=[1, 1, 1],
        src_offsets=[0, 2, 3],
        dst_origins=[0, 2, 3],
        dst_steps=[1],
    )
    exported = torch.export.export(model, (_values[:4],) * 3)
    edge = to_edge(exported, compile_config=cortex_m_edge_compile_config())
    [copy_node] = [
        node
        for node in edge.exported_program().graph.nodes
        if node.target == exir_ops.edge.cortex_m.strided_copy.default
    ]
    args = list(copy_node.args)
    args[5] = origins
    copy_node.args = tuple(args)
    edge.exported_program().graph_module.recompile()
    with pytest.raises((ValueError, RuntimeError), match=f"strided_copy: .*{message}"):
        edge.to_executorch()
