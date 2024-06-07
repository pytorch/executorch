# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Test first-stage conversion to TOSA within the Arm backend.
#

import copy
import unittest

import executorch.exir as exir
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.arm_partitioner import ArmPartitioner

## For quantization
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test.test_models import TestList, TosaProfile
from executorch.exir import EdgeCompileConfig
from executorch.exir.program import to_edge
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import export

# Config for Capturing the weights, will be moved in the future
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)
_EDGE_COMPILE_CONFIG: EdgeCompileConfig = exir.EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
)


class TestBasicNN(unittest.TestCase):
    def test_minimal_MI(self):
        for test_model in TestList:
            print(f"Running test {test_model}")
            model, inputs, outputs = prepare_model_and_ref(test_model, TosaProfile.MI)
            if inputs is None:
                print("  Skipping, no inputs for this profile")
                continue
            model_edge, exec_prog = export_model(
                model,
                inputs,
                ArmCompileSpecBuilder().tosa_compile_spec().build(),
            )

    def test_minimal_BI(self):
        for test_model in TestList:
            print(f"Running test {test_model}")
            model, inputs, outputs = prepare_model_and_ref(test_model, TosaProfile.BI)
            if inputs is None:
                print("  Skipping, no inputs for this profile")
                continue
            model_edge, exec_prog = export_model(
                model,
                inputs,
                ArmCompileSpecBuilder().tosa_compile_spec().build(),
            )

    def test_minimal_BI_INT(self):
        for test_model in TestList:
            print(f"Running test {test_model}")
            model, inputs, outputs = prepare_model_and_ref(
                test_model, TosaProfile.BI_INT
            )
            if inputs is None:
                print("  Skipping, no inputs for this profile")
                continue
            model_edge, exec_prog = export_model(
                model,
                inputs,
                ArmCompileSpecBuilder().tosa_compile_spec().build(),
            )


def prepare_model_and_ref(test_model, profile=TosaProfile.MI):
    model = TestList[test_model]
    model_inputs = model.inputs.get(profile)

    if model_inputs is None:
        return model, model_inputs, None

    model.eval()
    if profile == TosaProfile.BI:
        permute_memory_to_nhwc = model.permute_memory_to_nhwc
        # Quantize the model
        captured_model_graph_module = capture_pre_autograd_graph(
            model, copy.deepcopy(model.inputs[profile])
        )
        # Setup the quantizer
        quantizer = ArmQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)

        # Quantize
        prepared_model = prepare_pt2e(captured_model_graph_module, quantizer)
        prepared_model(*model.inputs[profile])
        model = convert_pt2e(prepared_model)
        model.permute_memory_to_nhwc = permute_memory_to_nhwc

    model_outputs = model.forward(*model_inputs)
    return model, model_inputs, model_outputs


def export_model(model, inputs, compile_spec):
    model_capture = export(model, inputs)
    model_edge = to_edge(model_capture, compile_config=_EDGE_COMPILE_CONFIG)

    model_edge = model_edge.to_backend(ArmPartitioner(compile_spec))
    exec_prog = model_edge.to_executorch()
    return model_edge, exec_prog
