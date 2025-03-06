import unittest

import executorch
import torch
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export, ExportedProgram


class BaseOpenvinoOpTest(unittest.TestCase):
    device = "CPU"

    atol = 1e-3
    rtol = 1e-3

    def execute_layer_test(
        self,
        module: torch.nn.Module,
        sample_inputs: tuple[torch.Tensor],
        expected_partitions: int = 1,
        assert_output_equal: bool = True,
    ):

        module = module.eval()
        # Export to aten dialect using torch.export
        aten_dialect: ExportedProgram = export(module, sample_inputs)

        # Convert to edge dialect and lower the module to the backend with a custom partitioner
        compile_spec = [CompileSpec("device", self.device.encode())]
        lowered_module: EdgeProgramManager = to_edge_transform_and_lower(
            aten_dialect,
            partitioner=[
                OpenvinoPartitioner(compile_spec),
            ],
        )

        # Apply backend-specific passes
        exec_prog = lowered_module.to_executorch(
            config=executorch.exir.ExecutorchBackendConfig()
        )

        # Check if the number of partitions created matches the expected number of partitions
        self.assertEqual(
            len(exec_prog.executorch_program.execution_plan[0].delegates),
            expected_partitions,
        )
        # Check if the individual partitions are assigned to Openvino backend
        for i in range(expected_partitions):
            self.assertEqual(
                exec_prog.executorch_program.execution_plan[0].delegates[i].id,
                OpenvinoBackend.__name__,
            )

        # Execute the model and compare the outputs with the reference outputs
        if assert_output_equal:
            # Execute the module in eager mode to calculate the reference outputs
            ref_output = module(*sample_inputs)
            if isinstance(ref_output, torch.Tensor):
                ref_output = [
                    ref_output,
                ]

            # Load model from buffer and execute
            executorch_module = _load_for_executorch_from_buffer(exec_prog.buffer)
            outputs = executorch_module.run_method("forward", sample_inputs)

            # Compare the outputs with the reference outputs
            self.assertTrue(len(ref_output) == len(outputs))
            for i in range(len(ref_output)):
                self.assertTrue(
                    torch.allclose(
                        outputs[i],
                        ref_output[i],
                        atol=self.atol,
                        rtol=self.rtol,
                        equal_nan=True,
                    ),
                    msg=f"ref_output:\n{ref_output[i]}\n\ntest_output:\n{outputs[i]}",
                )
