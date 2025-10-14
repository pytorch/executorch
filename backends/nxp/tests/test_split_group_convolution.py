# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import numpy as np
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.aten_passes.split_group_convolution import (
    SplitGroupConvolution,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.tests.executorch_pipeline import (
    _quantize_model,
    get_random_calibration_inputs,
    to_model_input_spec,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.models import (
    Conv1dModule,
    Conv2dModule,
    Conv3dModule,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.export_util import export_to_edge
from parameterized import parameterized
from torch.fx import GraphModule


def _quantize_and_lower_module(
    module: GraphModule, input_shape: tuple[int, ...], target="imxrt700"
) -> EdgeProgramManager:
    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
    quantizer = NeutronQuantizer()

    exir_program_aten__module_quant = _quantize_model(
        module, quantizer, calibration_inputs
    )

    edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_program_manager = export_to_edge(
        exir_program_aten__module_quant,
        calibration_inputs[0],
        edge_compile_config=edge_compile_config,
    )

    compile_spec = generate_neutron_compile_spec(target, "SDK_25_09")
    partitioner = NeutronPartitioner(compile_spec)
    return edge_program_manager.to_backend(partitioner)


class TestSplitGroupConvolution(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests.

    @classmethod
    def setUp(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    @parameterized.expand(
        [
            ["group = 2", [1, 16, 10, 10], 2],
            ["group = 3", [1, 24, 10, 10], 3],
            ["group = 8", [1, 8, 10, 10], 8],
        ]
    )
    def test_split_group_convolution__2d(self, _, input_shape: list[int], group: int):
        example_input = (torch.ones(input_shape),)

        module = Conv2dModule(
            bias=True,
            in_channels=input_shape[1],
            out_channels=8
            * group,  # Make sure the output channels are multiple of 8, so the `cat` can be delegated.
            group=group,
            stride=1,
        )
        graph_module = torch.export.export(module, example_input, strict=True).module()
        original_module = deepcopy(graph_module)

        modified_module = NeutronAtenPassManager([SplitGroupConvolution()])(
            graph_module
        ).graph_module

        # Make sure the fusion worked.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 5
        assert original_nodes[3].target == torch.ops.aten.conv2d.default
        assert original_nodes[3].args[-1] == group

        assert len(modified_nodes) == 4 + group * 4
        assert modified_nodes[1].target == torch.ops.aten.split.default
        for node in modified_nodes[2 + 3 * group : 4 + 3 * group]:
            assert node.target == torch.ops.aten.conv2d.default
            assert node.args[-1] == 1  # Groups.
        assert modified_nodes[-2].target == torch.ops.aten.cat.default

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2, atol=2.0e-7, rtol=1.9e-4)

        # Make sure the graph can be correctly quantized and lowered to edge.
        ep = _quantize_and_lower_module(
            modified_module, tuple(input_shape)
        ).exported_program()
        nodes = list(ep.graph.nodes)
        assert nodes[-5].name == "lowered_module_0"
        assert not graph_contains_any_of_ops(
            ep.graph,
            [exir_ops.edge.aten.convolution.default, exir_ops.edge.aten.cat.default],
        )

    @parameterized.expand(
        [
            ["group = 2", [1, 16, 10], 2],
            ["group = 3", [1, 24, 10], 3],
            ["group = 6", [1, 24, 10], 6],
        ]
    )
    def test_split_group_convolution__1d(self, _, input_shape: list[int], group: int):
        example_input = (torch.ones(input_shape),)

        module = Conv1dModule(
            bias=True,
            in_channels=input_shape[1],
            out_channels=8
            * group,  # Make sure the output channels are multiple of 8, so the `cat` can be delegated.
            group=group,
            stride=1,
        )
        graph_module = torch.export.export(module, example_input).module()
        original_module = deepcopy(graph_module)

        modified_module = NeutronAtenPassManager([SplitGroupConvolution()])(
            graph_module
        ).graph_module

        # Make sure the fusion worked.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 5
        assert original_nodes[3].target == torch.ops.aten.conv1d.default
        assert original_nodes[3].args[-1] == group

        assert len(modified_nodes) == 4 + group * 4
        assert modified_nodes[1].target == torch.ops.aten.split.default
        for node in modified_nodes[2 + 3 * group : 4 + 3 * group]:
            assert node.target == torch.ops.aten.conv1d.default
            assert node.args[-1] == 1  # Groups.
        assert modified_nodes[-2].target == torch.ops.aten.cat.default

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2, atol=2.0e-7)

        # Make sure the graph can be correctly quantized and lowered to edge.
        ep = _quantize_and_lower_module(
            modified_module, tuple(input_shape)
        ).exported_program()
        nodes = list(ep.graph.nodes)
        assert nodes[-5].name == "lowered_module_0"
        assert not graph_contains_any_of_ops(
            ep.graph,
            [exir_ops.edge.aten.convolution.default, exir_ops.edge.aten.cat.default],
        )

    @parameterized.expand(
        [
            ["group = 2", [1, 16, 10, 10, 10], 2],
        ]
    )
    def test_split_group_convolution__3d(self, _, input_shape: list[int], group: int):
        example_input = (torch.ones(input_shape),)

        module = Conv3dModule(
            bias=True,
            in_channels=input_shape[1],
            out_channels=8
            * group,  # Make sure the output channels are multiple of 8, so the `cat` can be delegated.
            group=group,
        )
        graph_module = torch.export.export(module, example_input).module()
        original_module = deepcopy(graph_module)

        modified_module = NeutronAtenPassManager([SplitGroupConvolution()])(
            graph_module
        ).graph_module

        # Verify that the pass has NOT made any changes, as it is disabled for 3D convolution.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == len(modified_nodes)
        for original_node, modified_node in zip(original_nodes, modified_nodes):
            assert original_node.name == modified_node.name
            assert original_node.target == modified_node.target

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2)

    def test_split_group_convolution__applied_by_default(self):
        input_shape = [1, 16, 10, 10]
        group = 2
        example_input = (torch.ones(input_shape),)

        module = Conv2dModule(
            in_channels=input_shape[1],
            out_channels=8
            * group,  # Make sure the output channels are multiple of 8, so the `cat` can be delegated.
            group=group,
            stride=1,
        )
        graph_module = torch.export.export(module, example_input).module()
        original_module = deepcopy(graph_module)

        modified_module = NeutronAtenPassManager()(
            graph_module
        ).graph_module  # Default passes.

        # Make sure the fusion worked.
        original_nodes = list(original_module.graph.nodes)
        modified_nodes = list(modified_module.graph.nodes)

        assert len(original_nodes) == 5
        assert original_nodes[3].target == torch.ops.aten.conv2d.default
        assert original_nodes[3].args[-1] == group

        assert len(modified_nodes) == 4 + group * 4
        assert modified_nodes[1].target == torch.ops.aten.split.default
        for node in modified_nodes[2 + 3 * group : 4 + 3 * group]:
            assert node.target == torch.ops.aten.conv2d.default
            assert node.args[-1] == 1  # Groups.
        assert modified_nodes[-2].target == torch.ops.aten.cat.default

        # Verify that the behavior has not changed.
        input_data = torch.randn(input_shape, dtype=torch.float32)
        out1 = original_module(input_data).detach().numpy()
        out2 = modified_module(input_data).detach().numpy()
        assert np.allclose(out1, out2, atol=5.0e-7)

        # Make sure the graph can be correctly quantized and lowered to edge.
        ep = _quantize_and_lower_module(
            modified_module, tuple(input_shape)
        ).exported_program()
        nodes = list(ep.graph.nodes)
        assert nodes[-5].name == "lowered_module_0"
        assert not graph_contains_any_of_ops(
            ep.graph,
            [exir_ops.edge.aten.convolution.default, exir_ops.edge.aten.cat.default],
        )
