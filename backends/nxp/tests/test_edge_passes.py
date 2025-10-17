# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import _is_dequantize, _is_quantize
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    EdgeProgramExecutor,
    OverrideTargetSupportCheck,
)
from executorch.backends.nxp.tests.models import (
    ConvActivationModule,
    ConvFCFCSoftmaxModuleWithoutReshape,
    LinearActivationModule,
)
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram
from torch.fx import Graph, Node


def _is_view_copy(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target == exir_ops.edge.aten.view_copy.default
    )


def _find_view_copy_node_indices(graph_nodes: list[Node]) -> list[int]:
    view_copy_nodes_indices = []

    for idx, node in enumerate(graph_nodes):
        if _is_view_copy(node):
            view_copy_nodes_indices.append(idx)

    return view_copy_nodes_indices


def _assert_nodes_form_a_view_copy_qdq_cluster(graph: Graph, node_indices: list[int]):
    assert len(node_indices) == 3

    nodes = list(graph.nodes)
    assert _is_dequantize(dequantize := nodes[node_indices[0]])
    assert _is_view_copy(view_copy := nodes[node_indices[1]])
    assert _is_quantize(quantize := nodes[node_indices[2]])

    # Make sure the nodes are properly connected.
    assert view_copy.args[0] == dequantize
    assert quantize.args[0] == view_copy


class TestEdgePasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    def test_moving_view_copy_into_separate_qdq_clusters(self):
        model = ConvFCFCSoftmaxModuleWithoutReshape()
        input_shape = (1, 4, 3, 33)

        # Prohibit `view_copy` conversion for the testing purposes.
        def unsupported_target(*_):
            return False

        with OverrideTargetSupportCheck(
            ViewCopyConverter, new_target_support_check=unsupported_target
        ):
            epm = to_quantized_edge_program(model, input_shape, target="imxrt700")
            exported_program = epm.exported_program()

            nodes = list(exported_program.graph_module.graph.nodes)
            assert len(nodes) == 28

            view_copy_indices = _find_view_copy_node_indices(nodes)

            assert len(view_copy_indices) == 4
            for idx in view_copy_indices:
                _assert_nodes_form_a_view_copy_qdq_cluster(
                    exported_program.graph, node_indices=[idx - 1, idx, idx + 1]
                )

            # Make sure the program is runnable.
            input_data = np.random.random(input_shape).astype("float32")
            program_executor = EdgeProgramExecutor(exported_program)
            program_executor.inference(input_data)

    @parameterized.expand(
        [
            ["relu"],
            ["relu6"],
            ["tanh"],
            ["sigmoid"],
        ]
    )
    def test_moving_fusable_activations_into_separate_qdq_clusters__addmm(
        self, activation
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:

            input_shape = (1, 4)
            model = LinearActivationModule(
                activation=activation,
                inplace=True,
                in_channels=input_shape[1],
                mode="addmm",
            )

            _ = to_quantized_edge_program(model, input_shape)
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            # Check linear and activation are in separate QDQ clusters
            nodes = list(exported_program.graph.nodes)
            assert len(nodes) == 12
            assert _is_dequantize(nodes[5])
            assert (
                neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__edge(
                    nodes[6]
                )
            )
            assert _is_quantize(nodes[7])
            assert _is_dequantize(nodes[8])
            assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__edge(
                nodes[9]
            )
            assert _is_quantize(nodes[10])

    @parameterized.expand(
        [
            ["relu"],
            ["relu6"],
            ["tanh"],
            ["sigmoid"],
        ]
    )
    def test_moving_fusable_activations_into_separate_qdq_clusters__mm(
        self, activation
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:

            input_shape = (1, 4)
            model = LinearActivationModule(
                activation=activation,
                inplace=True,
                in_channels=input_shape[1],
                mode="mm",
            )

            _ = to_quantized_edge_program(model, input_shape)
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            # Check linear and activation are in separate QDQ clusters
            nodes = list(exported_program.graph.nodes)
            assert len(nodes) == 10
            assert _is_dequantize(nodes[3])
            assert (
                neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__edge(
                    nodes[4]
                )
            )
            assert _is_quantize(nodes[5])
            assert _is_dequantize(nodes[6])
            assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__edge(
                nodes[7]
            )
            assert _is_quantize(nodes[8])

    @parameterized.expand(
        [
            ["relu"],
            ["relu6"],
            ["tanh"],
            ["sigmoid"],
        ]
    )
    def test_moving_fusable_activations_into_separate_qdq_clusters__linear(
        self, activation
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:

            input_shape = (1, 4)
            model = LinearActivationModule(
                activation=activation,
                inplace=True,
                in_channels=input_shape[1],
                mode="linear",
            )

            _ = to_quantized_edge_program(model, input_shape)
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            # Check linear and activation are in separate QDQ clusters
            nodes = list(exported_program.graph.nodes)
            assert len(nodes) == 13
            assert _is_dequantize(nodes[5])
            assert (
                neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__edge(
                    nodes[7]
                )
            )
            assert _is_quantize(nodes[8])
            assert _is_dequantize(nodes[9])
            assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__edge(
                nodes[10]
            )
            assert _is_quantize(nodes[11])

    @parameterized.expand(
        [
            ["relu"],
            ["relu6"],
            ["tanh"],
            ["sigmoid"],
        ]
    )
    def test_moving_fusable_activations_into_separate_qdq_clusters__conv(
        self, activation
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:

            input_shape = (1, 4, 8, 8)
            model = ConvActivationModule(
                activation=activation, inplace=True, in_channels=input_shape[1]
            )

            _ = to_quantized_edge_program(model, input_shape)
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            # Check linear and activation are in separate QDQ clusters
            nodes = list(exported_program.graph.nodes)
            assert len(nodes) == 16
            assert _is_dequantize(nodes[9])
            assert (
                neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__edge(
                    nodes[10]
                )
            )
            assert _is_quantize(nodes[11])
            assert _is_dequantize(nodes[12])
            assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__edge(
                nodes[13]
            )
            assert _is_quantize(nodes[14])
