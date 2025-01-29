import io
import json
import subprocess
import sys
import tempfile
import unittest
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

import torch
import executorch
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir.backend.backend_details import CompileSpec
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.exir.program import ExecutorchProgram, ExecutorchProgramManager
from executorch.backends.openvino.tests.models import *

import os
import random

from collections import defaultdict
from typing import List

import argparse

class TestOpenVINO(unittest.TestCase):
    device = "CPU"
    build_folder = None

    def execute_layer_test(
        self,
        module: torch.nn.Module,
        sample_inputs: tuple[torch.Tensor],
        expected_partitions: int = 1,
        assert_output_equal: bool = True,
    ):

        module = module.eval()
        aten_dialect: ExportedProgram = export(module, sample_inputs)

        edge_program: EdgeProgramManager = to_edge(aten_dialect)
        to_be_lowered_module = edge_program.exported_program()

        compile_spec = [CompileSpec("device", self.device.encode())]
        lowered_module = edge_program.to_backend(OpenvinoPartitioner(compile_spec))

        exec_prog = lowered_module.to_executorch(config=executorch.exir.ExecutorchBackendConfig())

        self.assertEqual(
            len(exec_prog.executorch_program.execution_plan[0].delegates),
            expected_partitions,
        )
        for i in range(expected_partitions):
            self.assertEqual(
                exec_prog.executorch_program.execution_plan[0].delegates[i].id,
                OpenvinoBackend.__name__,
            )

        if (assert_output_equal):
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_list = ""
                for idx, _ in enumerate(sample_inputs):
                    input_name = f"input_0_{idx}.raw"
                    input_list += input_name + " "
                input_list = input_list.strip() + "\n"

                output_dir = f"{tmp_dir}/outputs"

                ref_output = module(*sample_inputs)
                if isinstance(ref_output, torch.Tensor):
                    ref_output = [ref_output,]

                pte_fname = f"{tmp_dir}/openvino_executorch_test.pte"
                with open(pte_fname, "wb") as file:
                    exec_prog.write_to_file(file)


                self.generate_inputs(tmp_dir, "input_list.txt", [sample_inputs], input_list)
                self.make_output_dir(output_dir)

                cmd = [
                    # openvino_executor_runner
                    f"{self.build_folder}/examples/openvino/openvino_executor_runner",
                    pte_fname,
                    #"--input_list_path",
                    f"{tmp_dir}/input_list.txt",
                    #"--output_folder_path",
                    output_dir,
                    #"--method_index",
                    #str(method_index),
                ]

                env = dict(os.environ)
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=tmp_dir,
                )

                stdout_str = proc.stdout.decode('utf-8')
                self.assertIn("Model executed successfully.", stdout_str)

                output_dir = f"{tmp_dir}/outputs"
                outputs = []

                for i, f in enumerate(sorted(os.listdir(output_dir))):
                    filename = os.path.join(output_dir, f)
                    output = np.fromfile(filename, dtype=ref_output[i].numpy().dtype)
                    output = torch.from_numpy(output).reshape(ref_output[i].shape)
                    outputs.append(output)

                self.assertTrue(len(ref_output) == len(outputs))
                for i in range(len(ref_output)):
                    self.assertTrue(
                        torch.allclose(
                            outputs[i], ref_output[i], atol=self.atol, rtol=self.rtol
                        ),
                        msg=f"ref_output:\n{ref_output[i]}\n\ntest_output:\n{outputs[i]}",
                    )

    def generate_inputs(self, dest_path: str, file_name: str, inputs=None, input_list=None):
        input_list_file = None
        input_files = []
    
        # Prepare input list
        if input_list is not None:
            input_list_file = f"{dest_path}/{file_name}"
            with open(input_list_file, "w") as f:
                f.write(input_list)
                f.flush()
    
        # Prepare input data
        if inputs is not None:
            for idx, data in enumerate(inputs):
                for i, d in enumerate(data):
                    file_name = f"{dest_path}/input_{idx}_{i}.raw"
                    d.detach().numpy().tofile(file_name)
                    input_files.append(file_name)
    
        return input_list_file, input_files

    def make_output_dir(self, path: str):
        if os.path.exists(path):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.removedirs(path)
        os.makedirs(path)


class TestOpenVINOloatingPointOperator(TestOpenVINO):
    atol = 1e-1
    rtol = 1e-1

    def test_openvino_backend_arange(self):
        module = Arange(5)  # noqa: F405
        #sample_input = (torch.randn(5),)
        sample_input = (torch.ones(5),)
        self.execute_layer_test(module, sample_input)


def setup_environment():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="OpenVINO device to execute the model on",
        type=str,
        default="CPU",
    )

    args, ns_args = parser.parse_known_args(namespace=unittest)
    TestOpenVINO.device = args.device
    TestOpenVINO.build_folder = args.build_folder
    return sys.argv[:1] + ns_args

if __name__ == "__main__":
    ut_args = setup_environment()
    unittest.main(argv=ut_args)
