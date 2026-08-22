# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, is_parameter


class TestBuildersUtils(unittest.TestCase):
    """
    Graph-level tests for the parameter lookup helpers in
    ``backends/qualcomm/builders/utils.py``. No QNN SDK or device required.
    """

    def _exported_program_with_get_attr(self, const, val_dtype, graph=None):
        """
        Build an ExportedProgram holding ``const`` as an *unlifted* tensor
        attribute, plus a ``get_attr`` node referring to it.

        Passes that materialize new weights (e.g. while rewriting linear into
        conv2d) attach the tensor with ``setattr`` and emit a plain ``get_attr``
        instead of going through the lifted-constant machinery, so the node
        appears in neither ``state_dict`` nor ``constants`` and the export graph
        signature does not mention it.

        When ``graph`` is passed the node is created in that graph instead of the
        program's own graph, which leaves ``node.graph.owning_module`` unable to
        resolve the target.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        edge_program = torch.export.export(Model(), (torch.randn(2, 3),))
        graph_module = edge_program.graph_module
        attr_name = "_qnn_unlifted_const"
        setattr(graph_module, attr_name, const)

        target_graph = graph_module.graph if graph is None else graph
        node = target_graph.create_node("get_attr", attr_name, name=attr_name)
        node.meta["val"] = const.to(dtype=val_dtype, device="meta")
        return edge_program, node

    def test_is_parameter_accepts_unlifted_get_attr(self):
        const = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        edge_program, node = self._exported_program_with_get_attr(
            const, torch.int32
        )

        self.assertTrue(is_parameter(node, edge_program))

    def test_get_parameter_reads_unlifted_get_attr(self):
        const = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        edge_program, node = self._exported_program_with_get_attr(
            const, torch.int32
        )

        param = get_parameter(node, edge_program)

        self.assertIsInstance(param, torch.Tensor)
        # get_parameter casts to the QNN-qualified dtype recorded on the node
        self.assertEqual(param.dtype, torch.int32)
        self.assertTrue(torch.equal(param, const.to(torch.int32)))

    def test_get_parameter_falls_back_to_program_graph_module(self):
        # A node parked in a detached graph has no owning module to read the
        # attribute from; the lookup must fall back to the program's own module.
        const = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        edge_program, node = self._exported_program_with_get_attr(
            const, torch.float32, graph=torch.fx.Graph()
        )
        self.assertIsNone(node.graph.owning_module)

        self.assertTrue(is_parameter(node, edge_program))
        self.assertTrue(torch.equal(get_parameter(node, edge_program), const))


if __name__ == "__main__":
    unittest.main()
