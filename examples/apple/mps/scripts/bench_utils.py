#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import time
import unittest
from typing import Tuple

import torch
from torch.export.exported_program import ExportedProgram


class TestModule(unittest.TestCase):
    def assert_outputs_equal(self, model_output, ref_output, use_fp16: bool = False):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the MPS's backend, we relax the detal such that absolute tolerance is 1e-3. and
        relative tolerance is 1e-3.
        """
        # Compare the result from executor and eager mode direclty
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            assert len(ref_output) == len(
                model_output
            ), "Length of outputs is not matching!"
            for i in range(len(ref_output)):
                res_output = model_output[i].cpu()
                ref_output = ref_output[i].cpu()
                if use_fp16 and ref_output.dtype == torch.float16:
                    # cast back from fp16 to fp32 (ExecuTorch results are in FP32 by default)
                    ref_output = ref_output.to(torch.float32)

                mean_err = ((res_output - ref_output).abs() / ref_output).mean()
                logging.info(f"mean err = {mean_err}")
                self.assertLess(mean_err, 0.05)
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            if use_fp16 and ref_output.dtype == torch.float16:
                # cast back from fp16 to fp32 (ExecuTorch results are in FP32 by default)
                ref_output = ref_output.to(torch.float32)
            ref_output = ref_output.cpu()
            res_output = model_output[0].cpu()
            mean_err = ((res_output - ref_output).abs() / ref_output).mean()
            logging.info(f"mean err = {mean_err}")
            self.assertLess(mean_err, 0.05)


def bench_forward(func, *args):
    # warmup
    for _ in range(10):
        func(*args)

    start = time.time()
    for _ in range(100):
        func(*args)
    end = time.time()
    return end - start


def executorch_forward_pass(model, inputs):
    for _ in range(10):
        model.forward(inputs)


def synchronize():
    torch.mps.synchronize()


def pytorch_forward_pass(model, inputs):
    for _ in range(10):
        model(*inputs)
    synchronize()


def get_mps_inputs(inputs):
    inputs_mps = []
    for tensor in inputs:
        inputs_mps.append(tensor.to("mps"))
    inputs_mps = tuple(inputs_mps)
    return inputs_mps


def get_executorch_model(executorch_program: ExportedProgram):
    try:
        from executorch.extension.pybindings.portable_lib import (  # @manual
            _load_for_executorch_from_buffer,
        )

        return _load_for_executorch_from_buffer(executorch_program.buffer)
    except ImportError:
        logging.info(
            "ExecuTorch MPS delegate was built without pybind support (not possible to run forward pass within python)"
        )
        return None


def bench_torch(executorch_program: ExportedProgram, model, inputs, model_name):
    model = model.to("mps")
    inputs_mps = get_mps_inputs(inputs)

    executorch_model = get_executorch_model(executorch_program)
    if executorch_model is not None:
        t_pytorch = bench_forward(pytorch_forward_pass, model, inputs_mps)
        t_executorch = bench_forward(executorch_forward_pass, executorch_model, inputs)

        logging.info(f"Model name: {model_name}")
        logging.info(f"Pytorch MPS forward pass: {t_pytorch} seconds")
        logging.info(f"ExecuTorch MPS forward pass: {t_executorch} seconds")
        logging.info(
            f"ExecuTorch speedup: {((t_pytorch - t_executorch) / t_pytorch) * 100}%"
        )


def compare_outputs(
    executorch_program: ExportedProgram,
    model: torch.nn.Module,
    inputs: Tuple[torch.tensor],
    model_name: str,
    use_fp16: bool,
):
    test_module = TestModule()
    inputs_copy = []
    if use_fp16:
        model = model.to(torch.float16)
    model = model
    for t in inputs:
        tensor = t.detach().clone()
        if use_fp16 and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        inputs_copy.append(tensor)
    inputs_copy = tuple(inputs_copy)

    pytorch_results = model(*inputs_copy)

    executorch_model = get_executorch_model(executorch_program)
    if executorch_model is not None:
        executorch_results = executorch_model.forward(inputs)
        test_module.assert_outputs_equal(executorch_results, pytorch_results, use_fp16)
        logging.info(
            f"Results between ExecuTorch forward pass with MPS backend and PyTorch forward pass for {model_name} are matching!"
        )
