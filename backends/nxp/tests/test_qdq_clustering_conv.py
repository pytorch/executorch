# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dModule


def test_conv2d_partitioner():
    model = Conv2dModule(bias=False)

    edge_program = to_quantized_edge_program(model, (1, 4, 32, 32))

    # Get subgraph (module) that is delegated to neutron
    lowered_module = edge_program.exported_program().graph_module.lowered_module_0
    nodes = list(lowered_module.original_module.graph.nodes)

    assert len(nodes) == 7

    q_x_node = nodes[1]
    dq_w_node = nodes[2]
    dq_x_node = nodes[3]
    conv_node = nodes[4]
    q_y_node = nodes[5]

    assert "cluster" not in q_x_node.meta
    assert dq_w_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert dq_x_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert conv_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert q_y_node.meta["cluster"] == "aten_convolution_default_cluster"
