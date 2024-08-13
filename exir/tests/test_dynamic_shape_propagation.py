# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest import TestCase

from executorch import exir
from executorch.exir import to_edge
from executorch.exir.passes import DebugPass, HintBasedSymShapeEvalPass, SpecPropPass
from executorch.exir.tests.models import Repeat, TensorItem
from torch.export import export


class TestDynamicShapeProp(TestCase):
    def test_repeat(self):
        eager_model = Repeat()
        inputs = eager_model.get_random_inputs()
        inputs = inputs[0], inputs[1]

        prog = to_edge(
            export(eager_model, inputs, dynamic_shapes=eager_model.get_dynamic_shape()),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )

        new_prog = prog.transform([SpecPropPass(), HintBasedSymShapeEvalPass()])

        gm = new_prog.exported_program().graph_module

        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]
        self.assertEqual(len(speclist), 2)
        first_spec, second_spec = speclist

        self.assertTrue(first_spec.is_upper_bound_tensor)
        self.assertTrue(second_spec.is_upper_bound_tensor)
        self.assertEqual(first_spec.shape, [4, 5])


class TestUnbackedSymInt(TestCase):
    def test_unbacked_symint(self):
        eager_model = TensorItem()
        inputs = eager_model.get_random_inputs()
        inputs = inputs[0], inputs[1]

        prog = to_edge(
            export(eager_model, inputs, dynamic_shapes=None),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_prog = prog.transform([SpecPropPass(), HintBasedSymShapeEvalPass()])
        gm = new_prog.exported_program().graph_module

        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]
        self.assertEqual(len(speclist), 1)
        self.assertTrue(speclist[0].is_upper_bound_tensor)
        self.assertEqual(
            speclist[0].shape, [100, 100]
        )  # upper bound of TensorItem model
