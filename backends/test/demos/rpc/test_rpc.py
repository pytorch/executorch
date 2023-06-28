import unittest

import torch
from executorch import exir
from executorch.backends.backend_api import to_backend
from executorch.backends.test.demos.rpc.executor_backend_partitioner import (
    ExecutorBackendPartitioner,
)
from executorch.backends.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.backends.test.op_partitioner_demo import AddMulPartitionerDemo

# pyre-ignore[21]: Could not find module `executorch.pybindings.portable`.
from executorch.pybindings.portable import _load_for_executorch_from_buffer  # @manual
from torch.utils._pytree import tree_flatten

"""
Server can be an App Call, and will send delegate to client backend like DSP,
DSP will reeive the rpc call, calls the Executorch instance (like on DSP),
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



For example, in boltnn usecase, there are can be three layers AP -> DSP -> HTA

AP
——
1. Ap instantiate Executorch instance on AP with DSPBackend
2. In DSPBackend init/execute, it'll invoke the implemented RPC calls on DSP

DSP
——
3. DSP receives the RPC call and construct the Executorch instance on the DSP
4. When dsp executor runs, it can call any delegate (e.g. HTA) as needed.

There’ll negligible overhead in binary size on the AP, as the executor size is small.
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
        graph_module = (
            exir.capture(
                simple_net, simple_net_input, exir.CaptureConfig(pt2_mode=True)
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .graph_module
        )
        # delegate the whole graph to the client executor
        lowered_module = to_backend(ExecutorBackend.__name__, graph_module, [])

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered_module

            def forward(self, *args):
                return self.lowered(*args)

        composite_model = CompositeModule()

        exec_prog = (
            exir.capture(
                composite_model, simple_net_input, exir.CaptureConfig(pt2_mode=True)
            )
            .to_edge()
            .to_executorch()
        )

        executorch_module = _load_for_executorch_from_buffer(exec_prog.buffer)

        # Now client executor is instantiate
        inputs_flattened, _ = tree_flatten(simple_net_input)

        # Send the input from server executor to client executor, and receive the result from client executor
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = composite_model(*simple_net_input)

        # Compare the server executor final result with eager model
        self.assertTrue(
            torch.allclose(model_output[0], ref_output[0], rtol=1e-03, atol=1e-03)
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

        graph_module = (
            exir.capture(model, inputs, exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )

        # First lower to demo backend
        demo_backend_lowered = to_backend(graph_module, AddMulPartitionerDemo)

        # Then lower to executor backend
        executor_backend_lowered = to_backend(
            demo_backend_lowered, ExecutorBackendPartitioner
        )

        prog_buffer = exir.export_graph_module_to_executorch(executor_backend_lowered)
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
