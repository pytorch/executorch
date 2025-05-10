# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dModule


def test_operator_selector_mechanism():
    model = Conv2dModule(bias=False)
    input_shape = (1, 4, 32, 32)

    operators_not_to_delegate = ["aten::convolution"]

    edge_program_manager = to_quantized_edge_program(
        model, input_shape, operators_not_to_delegate=operators_not_to_delegate
    )

    exported_program = edge_program_manager.exported_program()

    for node in exported_program.graph.nodes:
        if node.name == "aten_convolution_default":
            assert "delegation_tag" not in node.meta
