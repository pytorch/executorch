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
from executorch.exir.backend.test.demos.rpc.executor_backend_partitioner import (
    ExecutorBackendPartitioner,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.utils._pytree import tree_flatten

"""
Server can be an App Call, and will send delegate to client backend like DSP,
DSP will reeive the rpc call, calls the ExecuTorch instance (like on DSP),
and return the result.


       +------------+                                         +--------------+
       | Host (CPU) |                                         | Client (DSP) |
       +------------+                                         +--------------+
              |                                                     |
              |                                                     |
              v                                                     |
+--------------------------------+                                  |
|Instatiate an Executor instance |                                  |
+--------------------------------+                                  |
              |                                                     |
              |                                                     v
              v               send init call with     +---------------------------------+
    +--------------------+    delegate                | Unwarp the delegate,            |
    |                    | -------------------------->| instatiate an Executor instance |
    |init execution plan |                            +---------------------------------+
    |                    |                                          |
    |                    |    finish init call           +----------v----------+
    +--------------------+<------------------------------| init execution plan |
              |                                          +---------------------+
              |                                                     |
              v                                                     |
        +------------+        send the execute call                 v
        |            |---------------------------------------> +---------+
        |  execute   |        receive the execute result       | execute |
        +------------+<--------------------------------------  +---------+
              |                                                     |
              |                                                     |
              |                                                     |
              |                                                     v
              v



For example, in some usecases, there are can be three layers MCU -> DSP -> AC

MCU
——
1. MCU instantiate ExecuTorch instance with DSPBackend
2. In DSPBackend init/execute, it'll invoke the implemented RPC calls on DSP

DSP
——
3. DSP receives the RPC call and construct the ExecuTorch instance on the DSP
4. When dsp executor runs, it can call any delegate (e.g. Accelerator) as needed.

There’ll negligible overhead in binary size on the MCU, as the executor size is small.
"""


class TestRPCDemos(unittest.TestCase):
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
        # This example shows how to delegate the whole simple net to the client executor, like on DSP
        # CPU -> delegate (simple net) -> DSP executor run simple net

        simple_net = self.get_a_simple_net()
        simple_net_input = simple_net.get_example_inputs()
        exported_program = to_edge(
            export(simple_net, simple_net_input),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        # delegate the whole graph to the client executor
        lowered_module = to_backend(
            ExecutorBackend.__name__, exported_program.exported_program(), []
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered_module

            def forward(self, *args):
                return self.lowered(*args)

        composite_model = CompositeModule()

        exec_prog = to_edge(export(composite_model, simple_net_input)).to_executorch()

        executorch_module = _load_for_executorch_from_buffer(exec_prog.buffer)

        # Now client executor is instantiate
        inputs_flattened, _ = tree_flatten(simple_net_input)

        # Send the input from server executor to client executor, and receive the result from client executor
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*simple_net_input)

        # Compare the server executor final result with eager model
        self.assertTrue(
            torch.allclose(model_output[0], ref_output, rtol=1e-03, atol=1e-03)
        )

    def test_delegate_partial_program(self):
        # CPU -> delegate (simple net) -> DSP executor run simple net
        # (TODO): input -> linear (delegated to dsp executor) -> sigmoid -> linear (delegated to dsp executor) -> output
        pass

    def test_delegate_program_with_nested_delegate(self):
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
            ExecutorBackendPartitioner()
        )

        prog_buffer = executor_backend_lowered.to_executorch()
        buffer = prog_buffer.buffer

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
