import unittest

# necessary to ensure the ops are registered
import executorch.kernels.prim_ops.executorch_prim_ops_registry

import torch

# This class tests whether we can generate correct code to register the prim ops into PyTorch runtime.
class TestCustomOps(unittest.TestCase):
    def setUp(self) -> None:
        self.x = 1
        self.y = 2

    def test_add_registered(self) -> None:
        out_1 = torch.ops.executorch_prim.add.int(self.x, self.y)
        self.assertEqual(out_1, 3)
