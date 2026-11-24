# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import Counter
from dataclasses import dataclass

import pytest
import torch
from executorch.backends.espressif.lowering import get_operator_library_report
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import EspTester
from executorch.examples.models.mlperf_tiny.deep_autoencoder import DeepAutoEncoderModel
from executorch.examples.models.mlperf_tiny.ds_cnn import DSCNNKWSModel
from executorch.examples.models.mlperf_tiny.mobilenet_v1_025 import MobileNetV1025Model
from executorch.examples.models.mlperf_tiny.resnet8 import ResNet8Model
from executorch.examples.models.model_base import EagerModelBase
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

_FQ = exir_ops.edge.fused_quant
_ESP = exir_ops.edge.esp_nn


@dataclass
class ModelCase:
    model_factory: type[EagerModelBase]
    ops_before_transforms: dict[EdgeOpOverload, int]
    ops_after_transforms: dict[EdgeOpOverload, int]
    portable_ops: dict[str, int]
    quantization_ops: dict[str, int]
    layout_ops: dict[str, int]


test_cases = {
    "deep_autoencoder": ModelCase(
        DeepAutoEncoderModel,
        {_FQ.linear.default: 10, _FQ.relu.default: 9},
        {_ESP.quantized_linear_int8.default: 10},
        {},
        quantization_ops={
            "quantized_decomposed::quantize_per_tensor": 1,
            "quantized_decomposed::dequantize_per_tensor": 1,
        },
        layout_ops={},
    ),
    "ds_cnn": ModelCase(
        DSCNNKWSModel,
        {
            _FQ.convolution.default: 9,
            _FQ.relu.default: 9,
            _FQ.avg_pool2d.default: 1,
            _FQ.linear.default: 1,
        },
        {
            _ESP.quantized_conv2d_int8.default: 5,
            _ESP.quantized_depthwise_conv2d_int8.default: 4,
            _ESP.quantized_linear_int8.default: 1,
        },
        {
            "aten::avg_pool2d": 1,
            "aten::constant_pad_nd": 1,
            "aten::_softmax": 1,
        },
        quantization_ops={
            "quantized_decomposed::quantize_per_tensor": 3,
            "quantized_decomposed::dequantize_per_tensor": 3,
        },
        layout_ops={"aten::view_copy": 2, "aten::permute_copy": 2},
    ),
    "resnet8": ModelCase(
        ResNet8Model,
        {
            _FQ.convolution.default: 9,
            _FQ.relu.default: 7,
            _FQ.add.default: 3,
            _FQ.linear.default: 1,
        },
        {
            _ESP.quantized_conv2d_int8.default: 9,
            _ESP.quantized_add_int8.default: 3,
            _ESP.quantized_linear_int8.default: 1,
        },
        {
            "aten::constant_pad_nd": 2,
            "aten::mean.dim": 1,
            "aten::_softmax": 1,
        },
        quantization_ops={
            "quantized_decomposed::quantize_per_tensor": 5,
            "quantized_decomposed::dequantize_per_tensor": 5,
        },
        layout_ops={"aten::view_copy": 1, "aten::permute_copy": 3},
    ),
    "mobilenet_v1_025": ModelCase(
        MobileNetV1025Model,
        {
            _FQ.convolution.default: 27,
            _FQ.relu.default: 27,
            _FQ.linear.default: 1,
        },
        {
            _ESP.quantized_conv2d_int8.default: 14,
            _ESP.quantized_depthwise_conv2d_int8.default: 13,
            _ESP.quantized_linear_int8.default: 1,
        },
        {
            "aten::constant_pad_nd": 5,
            "aten::mean.dim": 1,
            "aten::_softmax": 1,
        },
        quantization_ops={
            "quantized_decomposed::quantize_per_tensor": 7,
            "quantized_decomposed::dequantize_per_tensor": 7,
        },
        layout_ops={"aten::view_copy": 1, "aten::permute_copy": 2},
    ),
}


@pytest.fixture(params=test_cases)
@torch.no_grad()
def model_case(request):
    name = request.param
    case = test_cases[name]
    if (
        _ESP.quantized_conv2d_int8.default in case.ops_after_transforms
        and get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None
    ):
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        wrapper = case.model_factory()
        model = wrapper.get_eager_model()
        inputs = wrapper.get_example_inputs()
        calibration = [wrapper.get_example_inputs() for _ in range(4)]
    return name, case, EspTester(model, inputs, calibration)


@torch.no_grad()
def test_dialect_mlperf_tiny(model_case):
    name, case, tester = model_case
    tester.test_dialect(
        case.ops_before_transforms,
        case.ops_after_transforms,
        qtol=1 if name == "deep_autoencoder" else 0,
        atol=0 if name == "deep_autoencoder" else 1e-3,
        rtol=torch.finfo(torch.float32).eps,
    )
    edge = tester.get_artifact()
    report = get_operator_library_report(edge)
    counts = {
        group: Counter(op["target"] for op in report[group])
        for group in ("cpu_ops", "quantization_ops", "layout_ops")
    }
    esp_ops = Counter(
        node.target.name()
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function"
        and isinstance(node.target, EdgeOpOverload)
        and node.target.namespace == "esp_nn"
    )
    assert esp_ops == {
        op.name(): count for op, count in case.ops_after_transforms.items()
    }
    assert counts["cpu_ops"] == case.portable_ops
    assert counts["quantization_ops"] == case.quantization_ops
    assert counts["layout_ops"] == case.layout_ops
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    plan = program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    print(json.dumps({"model": name, "esp_nn_ops": esp_ops, **counts}, indent=2))


@pytest.mark.skipif(
    not os.environ.get("ESPRESSIF_BUILD_DIR"),
    reason="Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory",
)
@torch.no_grad()
def test_implementation_mlperf_tiny(model_case, pytestconfig):
    _, _, tester = model_case
    tester.test_implementation(
        pytestconfig.rootpath, atol=1e-6, rtol=1e-5, num_inferences=2
    )
