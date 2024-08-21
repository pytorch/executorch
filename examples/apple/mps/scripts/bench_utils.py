#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import time
from typing import Tuple

import torch
from executorch.backends.apple.mps.test.test_mps_utils import TestMPS
from torch.export.exported_program import ExportedProgram


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
    test_module = TestMPS()
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
