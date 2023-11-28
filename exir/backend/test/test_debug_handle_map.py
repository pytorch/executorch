# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.exir.tests.models as models

import torch
from executorch import exir
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.backend.test.qnn_backend_demo import QnnBackend
from executorch.exir.delegate import executorch_call_delegate
from hypothesis import given, settings, strategies as st


class TestBackendDebugHandle(unittest.TestCase):
    def test_add_mul_partitioner(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        ep = exir.capture(m, inputs, exir.CaptureConfig()).to_edge()
        executorch_prog = ep
        executorch_prog.exported_program = to_backend(
            ep.exported_program, AddMulPartitionerDemo()
        )
        lowered_nodes = [
            getattr(executorch_prog.exported_program.graph_module, node.target)
            for node in executorch_prog.exported_program.graph.nodes
            if node.op == "get_attr"
        ]
        for lowered_node in lowered_nodes:
            self.assertEqual(len(lowered_node.meta["debug_handle_map"]), 2)

        call_delegate_nodes = [
            node
            for node in executorch_prog.exported_program.graph.nodes
            if node.target == executorch_call_delegate
        ]

        for call_delegate_node in call_delegate_nodes:
            self.assertIsNotNone(call_delegate_node.meta["debug_handle"])

    @given(
        unlift=st.booleans(),  # verify both lifted and unlifted graph
    )
    @settings(deadline=500000)
    def test_lowered_the_whole_model(self, unlift):
        module_list = [
            models.Emformer(),
            models.Repeat(),
            models.ElementwiseAdd(),
            models.MLP(),
            models.ModelWithUnusedArg(),
        ]

        capture_config = (
            exir.CaptureConfig(enable_aot=True) if unlift else exir.CaptureConfig()
        )

        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False, _use_edge_ops=True
        )

        for model in module_list:
            model_inputs = model.get_random_inputs()

            edgeir_m = exir.capture(model, model_inputs, capture_config).to_edge(
                edge_compile_config
            )
            lowered_model = to_backend(
                QnnBackend.__name__, edgeir_m.exported_program, []
            )

            # QnnBackend compile all nodes as one node. The debug_handle_map will be like (1: (debug handle from all nodes))
            # Ensure there is only one debug identifier
            self.assertEqual(
                len(lowered_model.meta["debug_handle_map"].keys()),
                1,
            )

            all_debug_handles = list(lowered_model.meta["debug_handle_map"].values())[0]
            self.assertEqual(
                len(all_debug_handles),
                len(lowered_model.original_module.graph.nodes),
            )

            class ComposedModel(torch.nn.Module):
                def __init__(self, lowered_model):
                    super().__init__()
                    self.back_bone = lowered_model

                def forward(self, *args):
                    return self.back_bone(*args)

            edge = exir.capture(
                ComposedModel(lowered_model), model_inputs, capture_config
            ).to_edge(edge_compile_config)
            lowered_nodes = [
                getattr(edge.exported_program.graph_module, node.target)
                for node in edge.exported_program.graph.nodes
                if node.op == "get_attr"
            ]
            for lowered_node in lowered_nodes:
                self.assertEqual(
                    len(lowered_node.meta["debug_handle_map"].keys()),
                    1,
                )

                all_debug_handles = list(
                    lowered_node.meta["debug_handle_map"].values()
                )[0]
                self.assertEqual(
                    len(all_debug_handles),
                    len(lowered_node.original_module.graph.nodes),
                )
