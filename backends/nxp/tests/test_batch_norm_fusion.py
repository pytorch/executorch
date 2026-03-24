# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
import pytest
import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.backend.graph_utils import batch_norm_target_ops
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    AddMMConverter,
    MMConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.view_copy_converter import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    graph_contains_any_of_ops,
    OverrideTargetSupportCheck,
)
from executorch.backends.nxp.tests.models import (
    ConvBatchNormModule,
    LinearBatchNormModule,
)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_conv_fusing(bias: bool):
    input_shape = [2, 4, 6, 8]
    example_input = (torch.ones(*input_shape),)

    module = ConvBatchNormModule(bias, len(input_shape), 4)
    program = torch.export.export(module, example_input, strict=True)
    og_module = program.module()

    pm = NeutronAtenPassManager(neutron_target_spec)
    transformed_module = pm(deepcopy(program.module())).graph_module

    # Make sure the fusion worked.
    assert graph_contains_any_of_ops(program.graph, batch_norm_target_ops)

    assert not graph_contains_any_of_ops(
        transformed_module.graph, batch_norm_target_ops
    )

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = og_module(input_data).detach().numpy()
    out2 = transformed_module(input_data).detach().numpy()
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_linear_fusing(bias: bool):
    input_shape = (2, 4)
    example_input = (torch.ones(*input_shape),)

    module = LinearBatchNormModule(
        bias, 2, input_shape[-1], input_shape[1], input_shape[1]
    )
    program = torch.export.export(module, example_input, strict=True)
    og_module = program.module()

    pm = NeutronAtenPassManager(neutron_target_spec)
    graph_module_out = pm(deepcopy(program.module())).graph_module

    # Make sure the fusion worked.
    assert graph_contains_any_of_ops(
        graph_module_out.graph,
        [
            torch.ops.aten.addmm.default,
            torch.ops.aten.linear.default,
        ],
    )

    assert not graph_contains_any_of_ops(graph_module_out.graph, batch_norm_target_ops)

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = og_module(input_data).detach().numpy()
    out2 = graph_module_out(input_data).detach().numpy()
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_conv_fusing__full_pipeline__1d(bias: bool):
    input_shape = [4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 6)

    edge_program = to_quantized_edge_program(
        module, tuple(input_shape)
    ).exported_program()

    assert len(edge_program.graph.nodes) == 15
    assert not graph_contains_any_of_ops(edge_program.graph, batch_norm_target_ops)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_conv_fusing__full_pipeline__2d(bias: bool):
    input_shape = [1, 4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 4)

    edge_program = to_quantized_edge_program(
        module, tuple(input_shape)
    ).exported_program()

    assert len(edge_program.graph.nodes) == 7
    assert not graph_contains_any_of_ops(edge_program.graph, batch_norm_target_ops)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
def test_batch_norm_linear_fusing__full_pipeline(bias: bool):
    input_shape = (2, 4)
    module = LinearBatchNormModule(
        bias, 2, input_shape[-1], input_shape[1], input_shape[1]
    )

    # Don't delegate the Linear node, because there seems to be a bug with the NeutronConverter/NeutronPartitioner.
    #  But that doesn't affect the validity of this test.
    def unsupported_target(*_):  # Accept all input arguments and return `False`.
        return False

    with OverrideTargetSupportCheck(
        AddMMConverter, new_target_support_check=unsupported_target
    ):
        with OverrideTargetSupportCheck(
            MMConverter, new_target_support_check=unsupported_target
        ):
            with OverrideTargetSupportCheck(
                ViewCopyConverter, new_target_support_check=unsupported_target
            ):
                edge_program = to_quantized_edge_program(
                    module, tuple(input_shape)
                ).exported_program()

    assert len(edge_program.graph.nodes) == 12
    assert not graph_contains_any_of_ops(edge_program.graph, batch_norm_target_ops)


@pytest.mark.parametrize(
    "bias", [True, False], ids=lambda x: "Bias" if x else "No bias"
)
@pytest.mark.parametrize(
    "input_shape",
    [[4, 6, 8], [2, 4, 6, 8], [2, 4, 6, 8, 10]],
    ids=lambda x: f"{len(x)}D",
)
@pytest.mark.parametrize("use_qat", [False, True], ids=lambda x: "QAT" if x else "PTQ")
def test_batch_norm_linear_incompatible__full_pipeline(
    bias: bool, input_shape: list[int], use_qat: bool
):
    if not use_qat:
        pytest.skip("Fusion done by `prepare_pt2e` itself. This is a bug in TorchAO.")

    module = LinearBatchNormModule(
        bias, len(input_shape), input_shape[-1], input_shape[1], input_shape[1]
    )

    # Don't delegate the Linear node, because there seems to be a bug with the NeutronConverter/NeutronPartitioner.
    #  But that doesn't affect the validity of this test.
    def unsupported_target(*_):  # Accept all input arguments and return `False`.
        return False

    with OverrideTargetSupportCheck(
        AddMMConverter, new_target_support_check=unsupported_target
    ):
        with OverrideTargetSupportCheck(
            MMConverter, new_target_support_check=unsupported_target
        ):
            with OverrideTargetSupportCheck(
                ViewCopyConverter, new_target_support_check=unsupported_target
            ):
                edge_program = to_quantized_edge_program(
                    module, tuple(input_shape), use_qat=use_qat
                ).exported_program()

    expected_num_of_nodes = 20 if bias else 18
    assert len(edge_program.graph.nodes) == expected_num_of_nodes
    assert graph_contains_any_of_ops(edge_program.graph, batch_norm_target_ops)


def test_biasless_convbn_fusion_qat():
    input_shape = (1, 3, 3, 3)
    model = ConvBatchNormModule(
        bias=False,
        input_rank=len(input_shape),
        num_features=3,
    )

    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=True, use_neutron_for_format_conversion=False
    ).exported_program()

    assert graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.higher_order.executorch_call_delegate]
    )
