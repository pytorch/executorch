# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.exir.passes.constant_prop_pass as constant_prop_module
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch.fx import GraphModule


class _constant_prop_context:
    def __init__(self):
        self.backup = constant_prop_module._DEFAULT_SKIP_TARGETS

    def __enter__(self):
        constant_prop_module._DEFAULT_SKIP_TARGETS = (
            constant_prop_module._DEFAULT_SKIP_TARGETS_NO_QUANT
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        constant_prop_module._DEFAULT_SKIP_TARGETS = self.backup


class ConstantPropPass(ExportPass):
    """
    Official constant_prop_pass will not fold Q-DQ
    But we need to fold quantized constant tensor as well as non-quantized one
    """

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def call(self, graph_module: GraphModule):
        with _constant_prop_context():
            _ = constant_prop_pass(self.edge_program)
        return PassResult(graph_module, True)
