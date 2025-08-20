import numpy as np
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    EdgeProgramExecutor,
    OverrideTargetSupportCheck,
)
from executorch.backends.nxp.tests.models import ConvFCFCSoftmaxModuleWithoutReshape
from executorch.exir.dialects._ops import ops as exir_ops
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


def test_moving_view_copy_into_separate_qdq_clusters():
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
