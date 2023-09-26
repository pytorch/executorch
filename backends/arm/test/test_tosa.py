# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Test first-stage conversion to TOSA within the Arm backend.
#

import unittest

import executorch.exir as exir
from executorch.backends.arm.arm_backend import ArmPartitioner
from executorch.backends.arm.test.test_models import TestList, TosaProfile

from executorch.exir.backend.backend_api import to_backend

# Config for Capturing the weights, will be moved in the future
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig()


class TestBasicNN(unittest.TestCase):
    def test_minimal_MI(self):
        for test_model in TestList:
            print(f"Running test {test_model}")
            model, inputs, outputs = prepare_model_and_ref(test_model, TosaProfile.MI)
            if inputs is None:
                print("  Skipping, no inputs for this profile")
                continue
            model_edge, exec_prog = export_model(model, inputs, [])
            # TODO: check there is a tosa delegate blob in the output

    def test_minimal_BI(self):
        for test_model in TestList:
            print(f"Running test {test_model}")
            model, inputs, outputs = prepare_model_and_ref(test_model, TosaProfile.BI)
            if inputs is None:
                print("  Skipping, no inputs for this profile")
                continue
            model_edge, exec_prog = export_model(model, inputs, [])
            # TODO: check there is a tosa delegate blob in the output


def prepare_model_and_ref(test_model, profile=TosaProfile.MI):
    model = TestList[test_model]
    model_inputs = model.inputs.get(profile)
    if model_inputs is not None:
        model_outputs = model.forward(*model_inputs)
        return model, model_inputs, model_outputs
    return model, model_inputs, None


def export_model(model, inputs, compile_spec):
    model_capture = exir.capture(model, inputs, _CAPTURE_CONFIG)
    model_edge = model_capture.to_edge(_EDGE_COMPILE_CONFIG)
    ArmPartitioner.compile_spec = compile_spec

    model_edge.exported_program = to_backend(
        model_edge.exported_program, ArmPartitioner
    )
    exec_prog = model_edge.to_executorch()
    return model_edge, exec_prog
