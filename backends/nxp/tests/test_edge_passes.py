# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import numpy as np
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    ViewCopyConverter,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_additional_quantize_dequantize_nodes_pass import (
    RemoveAdditionalQDQClustersPass,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.tests.executorch_pipeline import (
    _quantize_model,
    get_random_calibration_inputs,
    to_model_input_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    compare_output_arrays,
    EdgeProgramExecutor,
    OverrideTargetSupportCheck,
)
from executorch.backends.nxp.tests.ir.converter.node_converter.test_permute_copy_converter import (
    Conv2dPermuteModule,
)
from executorch.backends.nxp.tests.models import ConvFCFCSoftmaxModuleWithoutReshape
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.export_util.utils import export_to_edge
from torch.fx import Graph, Node


def _is_view_copy(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target == exir_ops.edge.aten.view_copy.default
    )


def _is_dequantize(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target.__name__
        == "quantized_decomposed.dequantize_per_tensor.default"
    )


def _is_quantize(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target.__name__ == "quantized_decomposed.quantize_per_tensor.default"
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

        # Prohibit `view_copy` conversion for the testing purposes.
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

    def test_remove_additional_quantize_dequantize_nodes_pass(self):
        input_shape = (1, 3, 8, 16)
        new_dims = (3, 2, 1, 0)
        model = Conv2dPermuteModule(input_shape[1], new_dims)
        target = "imxrt700"
        custom_delegation_options = CustomDelegationOptions()

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )

        example_input = calibration_inputs[0]
        exir_program_aten = torch.export.export(model, example_input).module()

        # Run pre-processing passes of the float32 aten dialect program.
        exir_program_aten = NeutronAtenPassManager()(exir_program_aten).graph_module

        exir_program_aten_quant = _quantize_model(
            exir_program_aten, NeutronQuantizer(), calibration_inputs
        )
        edge_program_manager = export_to_edge(
            exir_program_aten_quant,
            example_input,
        )

        edge_program_manager = NeutronEdgePassManager()(edge_program_manager)

        compile_spec = generate_neutron_compile_spec(target, "SDK_25_09")
        partitioner = NeutronPartitioner(compile_spec, custom_delegation_options)

        edge_program_manager = edge_program_manager.to_backend(partitioner)

        # Make sure QDQ cluster for permute_copy is present.
        edge_program_with_qdq_cluster = copy.deepcopy(
            edge_program_manager.exported_program()
        )
        nodes = list(edge_program_with_qdq_cluster.graph.nodes)
        assert len(nodes) == 10
        assert (
            nodes[5].target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        )
        assert nodes[6].target == exir_ops.edge.aten.permute_copy.default
        assert "cluster" in nodes[6].meta
        assert (
            nodes[7].target
            == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        )

        # Run pass for removal of additional QDQ nodes and compute in non-float types where possible
        edge_program_manager = NeutronEdgePassManager(
            [RemoveAdditionalQDQClustersPass()]
        )(edge_program_manager)

        # Make sure QDQ cluster for permute_copy is removed.
        edge_program_without_qdq_cluster = edge_program_manager.exported_program()
        nodes = list(edge_program_without_qdq_cluster.graph.nodes)
        assert len(nodes) == 8
        assert nodes[4].name == "getitem"
        assert nodes[5].target == exir_ops.edge.aten.permute_copy.default
        assert "cluster" not in nodes[5].meta
        assert (
            nodes[6].target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        )

        edge_program_executor_without_qdq_cluster = EdgeProgramExecutor(
            edge_program_without_qdq_cluster
        )
        edge_program_executor_with_qdq_cluster = EdgeProgramExecutor(
            edge_program_with_qdq_cluster
        )

        input_data = np.random.random(input_shape).astype(np.float32)
        edge_program_output_without_qdq_cluster = (
            edge_program_executor_without_qdq_cluster.inference(input_data)
        )
        edge_program_output_with_qdq_cluster = (
            edge_program_executor_with_qdq_cluster.inference(input_data)
        )

        compare_output_arrays(
            edge_program_output_without_qdq_cluster,
            edge_program_output_with_qdq_cluster,
            "main output",
        )
