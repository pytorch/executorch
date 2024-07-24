#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
import time

import torch
from torch.export.exported_program import ExportedProgram


def assert_outputs_equal(model_output, ref_output):
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
            assert torch.allclose(
                model_output[i], ref_output[i], atol=1e-03, rtol=1e-03
            )
    else:
        # If one output, eager returns tensor while executor tuple of size 1
        assert torch.allclose(
            model_output[0], ref_output, atol=1e-03, rtol=1e-03
        ), "Outputs are not matching!"


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


def compare_outputs(executorch_program: ExportedProgram, model, inputs, model_name):
    inputs_copy = []
    for t in inputs:
        inputs_copy.append(t.detach().clone())
    inputs_copy = tuple(inputs_copy)

    pytorch_results = model(*inputs)
    executorch_model = get_executorch_model(executorch_program)
    if executorch_model is not None:
        executorch_results = executorch_model.forward(inputs_copy)
        assert_outputs_equal(executorch_results, pytorch_results)
        logging.info(
            f"Results between ExecuTorch forward pass with MPS backend and PyTorch forward pass for {model_name} are matching!"
        )
