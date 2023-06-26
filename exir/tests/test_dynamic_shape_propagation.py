import copy
from unittest import TestCase

from executorch import exir
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import DebugPass, SpecPropPass, SymShapeEvalPass
from executorch.exir.passes.sym_shape_eval_pass import SymShapeEvalPass
from executorch.exir.tests.models import Repeat


class TestDynamicShapeProp(TestCase):
    def test_repeat(self):
        eager_model = Repeat()
        inputs = eager_model.get_random_inputs()
        inputs = inputs[0], inputs[1]

        prog = exir.capture(
            eager_model,
            inputs,
            exir.CaptureConfig(
                pt2_mode=True,
                enable_dynamic_shape=True,
            ),
        ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))

        new_prog = prog.transform(SpecPropPass(), SymShapeEvalPass())

        gm = new_prog.graph_module
        # meta is preserved thru deepcopy
        cp = copy.deepcopy(gm)
        self.assertTrue(len(cp.meta) > 0)

        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]
        self.assertEqual(len(speclist), 2)
        first_spec, second_spec = speclist

        self.assertTrue(first_spec.is_upper_bound_tensor)
        self.assertTrue(second_spec.is_upper_bound_tensor)
        self.assertEqual(first_spec.shape, [4, 5])
