# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch

from ..serialization.xts3 import parse_graph

from . import esp_nn_reference as reference


def _is_dequantize(node) -> bool:
    return node.op == "call_function" and "dequantize_per_tensor" in str(node.target)


def delegate_output_dtypes(lowered_module) -> list[torch.dtype]:
    program = lowered_module.original_module
    output = next(node for node in program.graph.nodes if node.op == "output")
    values = output.args[0]
    if not isinstance(values, (list, tuple)):
        values = (values,)
    return [torch.float32 if _is_dequantize(value) else torch.int8 for value in values]


class DelegateEmulator:
    """Runs one ESP32 int8 delegate payload exactly as the firmware backend does."""

    def __init__(self, lowered_module):
        self.graph = parse_graph(bytes(lowered_module.processed_bytes))
        self.interpreter = reference.ReferenceInterpreter(self.graph)
        self.output_dtypes = delegate_output_dtypes(lowered_module)

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        device_inputs = []
        for tensor_id, value in zip(self.interpreter.inputs, inputs):
            tensor = self.graph.tensors[tensor_id]
            array = value.detach().cpu().numpy()
            if value.dtype == torch.float32:
                array = reference.quantize(tensor, array)
            elif value.dtype != torch.int8:
                raise TypeError(
                    f"Delegate input must be float32 or int8, got {value.dtype}"
                )
            device_inputs.append(
                reference.to_device_layout(tensor, array.astype(np.int8))
            )
        outputs = self.interpreter.run_outputs(device_inputs)
        results = []
        for tensor_id, value, dtype in zip(
            self.interpreter.outputs, outputs, self.output_dtypes
        ):
            tensor = self.graph.tensors[tensor_id]
            logical = reference.to_logical_layout(tensor, value)
            if dtype == torch.float32:
                results.append(torch.from_numpy(reference.dequantize(tensor, logical)))
            else:
                results.append(torch.from_numpy(logical.astype(np.int8)))
        return tuple(results)


class ProgramEmulator(torch.fx.Interpreter):
    """
    Host model of an ExecuTorch program lowered with EspNNPartitioner:
    delegates run on the bit-exact ESP-NN reference, every other operator
    runs in PyTorch (the device uses the ExecuTorch portable kernels).
    """

    def __init__(self, exported_program):
        module = exported_program.module()
        super().__init__(module)
        self.delegates: dict[str, DelegateEmulator] = {}
        self.delegate_calls = 0
        self.cpu_calls = 0

    def call_function(self, target, args, kwargs):
        if "executorch_call_delegate" in str(target):
            lowered = args[0]
            key = id(lowered)
            if key not in self.delegates:
                self.delegates[key] = DelegateEmulator(lowered)
            self.delegate_calls += 1
            return self.delegates[key](*args[1:])
        if "getitem" not in str(target):
            self.cpu_calls += 1
        return super().call_function(target, args, kwargs)

    def __call__(self, *inputs):
        return self.run(*inputs)


__all__ = ["DelegateEmulator", "ProgramEmulator", "delegate_output_dtypes"]
