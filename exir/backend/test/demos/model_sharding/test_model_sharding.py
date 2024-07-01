# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.test.demos.model_sharding.executor_sharded_backend_partitioner import (
    ExecutorShardedBackendPartitioner,
)
from executorch.exir.backend.test.demos.model_sharding.executor_sharded_backend_preprocess import (
    ExecutorShardedBackend,
)
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.utils._pytree import tree_flatten


class TestModelShardingDemos(unittest.TestCase):
    def get_a_simple_net(self) -> torch.nn.Module:
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 25)
                self.linear2 = torch.nn.Linear(25, 3)

            def forward(self, x):
                x = torch.sigmoid(self.linear1(x))
                x = self.linear2(x)
                return x

            def get_example_inputs(self):
                return (torch.randn(25, 4),)

        return Net()

    def test_delegate_whole_program(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                a = x - b  # compute in dsp
                y = torch.mm(a, x)  # compute in hta
                z = y + b  # compute in hta
                return z

        model = Model()
        inputs = (torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))

        exported_program = to_edge(export(model, inputs))

        # First lower to demo backend
        demo_backend_lowered = exported_program.to_backend(AddMulPartitionerDemo())

        # Then lower to executor backend
        executor_backend_lowered = demo_backend_lowered.to_backend(
            ExecutorShardedBackendPartitioner()
        )

        prog_buffer = executor_backend_lowered.to_executorch()
        buffer = prog_buffer.buffer

        with open(
            "/data/users/chenlai/fbsource/fbcode/executorch/exir/backend/test/demos/model_sharding/shard.pte",
            "wb",
        ) as f:
            f.write(buffer)

        executorch_module = _load_for_executorch_from_buffer(buffer)

        # Now client executor is instantiate
        inputs_flattened, _ = tree_flatten(inputs)

        # Send the input from server executor to client executor, and receive the result from client executor
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = model(*inputs)

        # Compare the server executor final result with eager model
        self.assertTrue(
            torch.allclose(model_output[0], ref_output, rtol=1e-03, atol=1e-03)
        )
