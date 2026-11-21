# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod

import pytest
import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.passes.aten_to_cortex_m_pass import AtenToCortexMPass
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.test.tester import (
    CortexMRunPasses,
    CortexMTester,
    McuTestCase,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class CortexMCat(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 0,
        "executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default": 1,
    }
    qtol = 0

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class CortexMRepeatedCat(CortexMCat):
    def forward(self, x):
        return torch.cat((x, x, x, x), dim=self.dim)


class CortexMConstantCat(CortexMCat):
    def __init__(self, transpose_constant=False):
        super().__init__(1)
        constant = torch.arange(8, dtype=torch.int8).reshape(1, 2, 2, 2)
        if transpose_constant:
            constant = constant.transpose(2, 3)
        self.register_buffer("constant", constant)

    def forward(self, x):
        return torch.cat((x, self.constant), dim=self.dim)


class CortexMNestedCat(CortexMCat):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 2,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_aten_cat_default": 0,
        "executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default": 2,
    }

    def forward(self, x, y, z):
        return torch.cat((torch.cat((x, y), dim=self.dim), z), dim=self.dim)


class CortexMCatAfterAdd(CortexMCat):
    qtol = 1

    def forward(self, x, y):
        return torch.cat((x + x, y + y), dim=self.dim)


def _input(shape, channels_last=False):
    value = torch.arange(prod(shape), dtype=torch.int8).reshape(shape)
    if channels_last:
        value = value.to(memory_format=torch.channels_last)
    return value


test_cases = {
    "plain_quantized": McuTestCase(CortexMCat(), (torch.ones(2, 3), torch.zeros(1, 3))),
    "rank1": McuTestCase(CortexMCat(), (_input((3,)), _input((5,)))),
    "rank2_outer": McuTestCase(CortexMCat(), (_input((2, 3)), _input((4, 3)))),
    "rank2_negative_dim": McuTestCase(CortexMCat(-1), (_input((2, 3)), _input((2, 2)))),
    "rank3": McuTestCase(CortexMCat(1), (_input((2, 3, 4)), _input((2, 2, 4)))),
    "batch": McuTestCase(CortexMCat(), (_input((1, 3, 2, 4)), _input((2, 3, 2, 4)))),
    "channels": McuTestCase(
        CortexMCat(1), (_input((2, 2, 3, 4)), _input((2, 3, 3, 4)))
    ),
    "height": McuTestCase(CortexMCat(2), (_input((1, 2, 2, 4)), _input((1, 2, 3, 4)))),
    "width": McuTestCase(CortexMCat(3), (_input((1, 2, 3, 2)), _input((1, 2, 3, 4)))),
    "channels_last": McuTestCase(
        CortexMCat(1), (_input((2, 2, 3, 4), True), _input((2, 3, 3, 4), True))
    ),
    "mixed_layouts": McuTestCase(
        CortexMCat(1), (_input((2, 2, 3, 4)), _input((2, 3, 3, 4), True))
    ),
    "mixed_layouts_reversed": McuTestCase(
        CortexMCat(1), (_input((2, 2, 3, 4), True), _input((2, 3, 3, 4)))
    ),
    "singleton_channels_last": McuTestCase(
        CortexMCat(1), (_input((1, 1, 2, 3), True), _input((1, 2, 2, 3), True))
    ),
    "single_input": McuTestCase(CortexMCat(1), (_input((1, 2, 3, 4)),)),
    "repeated": McuTestCase(CortexMRepeatedCat(1), (_input((1, 2, 3, 4)),)),
    "constant": McuTestCase(CortexMConstantCat(), (_input((1, 3, 2, 2), True),)),
    "transposed_constant": McuTestCase(
        CortexMConstantCat(transpose_constant=True), (_input((1, 3, 2, 2), True),)
    ),
    "nested": McuTestCase(
        CortexMNestedCat(1),
        (_input((1, 1, 2, 3)), _input((1, 2, 2, 3)), _input((1, 3, 2, 3))),
    ),
    "empty_input": McuTestCase(CortexMCat(1), (_input((2, 0, 3)), _input((2, 2, 3)))),
    "empty_rank1_input": McuTestCase(CortexMCat(1), (_input((0,)), _input((2, 3)))),
    "quantized": McuTestCase(
        CortexMCatAfterAdd(1),
        (_input((2, 3, 4)).float() / 64, _input((2, 2, 4)).float() / 64),
    ),
    "quantized_channels_last": McuTestCase(
        CortexMCatAfterAdd(1),
        (
            _input((1, 3, 2, 4), True).float() / 64,
            _input((1, 2, 2, 4), True).float() / 64,
        ),
    ),
}

fallback_test_cases = {
    "int16": McuTestCase(
        CortexMCat(), (_input((2, 3)).short(), _input((1, 3)).short())
    ),
    "rank5": McuTestCase(
        CortexMCat(), (_input((1, 2, 3, 4, 5)), _input((1, 2, 3, 4, 5)))
    ),
    "strided_source": McuTestCase(
        CortexMCat(1), (_input((2, 6))[:, ::2], _input((2, 3)))
    ),
    "large_singleton_stride": McuTestCase(
        CortexMCat(1),
        (
            torch.empty_strided((1, 3), (2**31, 1), dtype=torch.int8).fill_(4),
            _input((1, 3)),
        ),
    ),
}


@parametrize("test_case", test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_cat(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        atol=0,
        qtol=test_case.model.qtol,
        use_explicit_layout=use_explicit_layout,
    )
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer


@parametrize("test_case", test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_implementation_cat(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(
        atol=0, qtol=test_case.model.qtol, use_explicit_layout=use_explicit_layout
    )


@parametrize("test_case", fallback_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_cat_fallback(test_case, use_explicit_layout, cortex_m_target):
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


@parametrize("test_case", fallback_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_implementation_cat_fallback(test_case, use_explicit_layout, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(atol=0, use_explicit_layout=use_explicit_layout)


@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_cat_float_fallback(use_explicit_layout, cortex_m_target):
    inputs = (torch.arange(6).float().reshape(2, 3), -torch.ones(1, 3))
    tester = CortexMTester(CortexMCat(), inputs, target_config=cortex_m_target)
    tester.export().to_edge().run_passes(
        CortexMRunPasses(
            target_config=cortex_m_target, use_explicit_layout=use_explicit_layout
        )
    )
    tester.check_count(CortexMCat.ops_before_transforms)
    tester.check_not(
        ["executorch_exir_dialects_edge__ops_cortex_m_strided_copy_default"]
    )
    tester.run_method_and_compare_outputs(inputs=inputs, atol=0)
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer


@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_cat_dynamic_fallback(use_explicit_layout):
    model = CortexMCat(1)
    batch = torch.export.Dim("batch", min=2, max=5)
    inputs = (_input((3, 2)), _input((3, 4)))
    exported = torch.export.export(
        model, inputs, dynamic_shapes={"inputs": ({0: batch}, {0: batch})}
    )
    edge = to_edge(exported, compile_config=cortex_m_edge_compile_config()).transform(
        CortexMPassManager(use_explicit_layout=use_explicit_layout)
    )
    assert not any(
        node.target == exir_ops.edge.cortex_m.strided_copy.default
        for node in edge.exported_program().graph.nodes
    )
    for size in (2, 4, 5):
        args = (_input((size, 2)), _input((size, 4)))
        torch.testing.assert_close(
            edge.exported_program().module()(*args), model(*args)
        )
    assert edge.to_executorch().buffer


@pytest.mark.parametrize("target", ["input", "output"])
@pytest.mark.parametrize("scale,zp", [(0.5, 0), (1.0, 1)])
def test_cat_rejects_mismatched_qparams(target, scale, zp, cortex_m_target):
    edge = to_edge(
        torch.export.export(CortexMCat(), (_input((2, 3)), _input((1, 3)))),
        compile_config=cortex_m_edge_compile_config(),
    )
    [cat] = [
        node
        for node in edge.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.cat.default
    ]
    qparams = QuantArgs(1.0, 0, -128, 127, torch.int8)
    cat.meta["input_qparams"] = {0: qparams}
    cat.meta["output_qparams"] = {0: qparams}
    changed = cat if target == "output" else cat.args[0][1]
    changed.meta["output_qparams"] = {0: qparams._replace(scale=scale, zp=zp)}
    with pytest.raises(
        ValueError, match="matching per-tensor input and output qparams"
    ):
        AtenToCortexMPass(edge.exported_program(), cortex_m_target).call(
            edge.exported_program().graph_module
        )
