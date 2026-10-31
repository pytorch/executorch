import os
import tempfile

import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.runtime import Runtime
from torch.export import Dim, export

op_params = [
    {"input_shape": [2, 3, 2], "target_shape": [2, 6]},
    {"input_shape": [4], "target_shape": [2, 2]},
]


class TestViewOperator(BaseOpenvinoOpTest):

    def create_model(self, target_shape):

        class View(torch.nn.Module):

            def __init__(self, target_shape) -> None:
                super().__init__()
                self.target_shape = target_shape

            def forward(self, input_tensor):
                return input_tensor.view(self.target_shape)

        return View(target_shape)

    def test_view(self):
        for params in op_params:
            with self.subTest(params=params):

                module = self.create_model(params["target_shape"])

                sample_input = (torch.randn(params["input_shape"]),)

                self.execute_layer_test(module, sample_input)

    def test_view_data_dependent_size(self):
        class View(torch.nn.Module):
            def __init__(self, reduce):
                super().__init__()
                self.reduce = reduce

            def forward(self, x, n):
                n = n.item()
                torch._check(n >= 1)
                torch._check(n <= x.shape[0])
                y = x[:n].view(n, -1)
                return y.sum(0) if self.reduce else y

        # The size reaches the OpenVINO partition as a SymInt input. The second
        # call needs a larger output than the first one returned, unless the
        # sum makes the output static.
        x = torch.randn(4, 8)
        for reduce in (False, True):
            with self.subTest(reduce=reduce):
                self.execute_layer_test(
                    View(reduce),
                    (x, torch.tensor(3)),
                    more_inputs=((x, torch.tensor(4)),),
                )

    def test_one_delegate_called_with_different_output_sizes(self):
        class Double(torch.nn.Module):
            def forward(self, x):
                return x * 2

        x, y = torch.randn(2, 8), torch.randn(4, 8)
        double = to_backend(
            OpenvinoBackend.__name__,
            to_edge(
                export(Double(), (y,), dynamic_shapes=({0: Dim("n", min=1, max=4)},))
            ).exported_program(),
            [CompileSpec("device", self.device.encode())],
        )
        # Programs from ExecuTorch 0.7 and earlier run every call of a lowered
        # module on one delegate handle, each call with its own outputs.
        double._share_handle_across_calls = True

        class TwoCalls(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = double

            def forward(self, x, y):
                return self.lowered(x), self.lowered(y)

        dynamic_shapes = (
            {0: Dim("rows_x", min=1, max=2)},
            {0: Dim("rows_y", min=1, max=4)},
        )
        program = to_edge(
            export(TwoCalls(), (x, y), dynamic_shapes=dynamic_shapes)
        ).to_executorch()
        self.assertEqual(len(program.executorch_program.execution_plan[0].delegates), 1)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            pte_path = os.path.join(tmp_dir, "model.pte")
            with open(pte_path, "wb") as f:
                program.write_to_file(f)
            loaded = Runtime.get().load_program(pte_path)
            method = loaded.load_method("forward")
            assert method is not None
            for inputs in ((x, y), (y[:1], y[:3])):
                outputs = method.execute(inputs)
                for output, value in zip(outputs, inputs):
                    torch.testing.assert_close(output, value * 2)
            del method, loaded
