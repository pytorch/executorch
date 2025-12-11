# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from executorch.backends.arm.tosa import TosaSpecification
from torchvision import models, transforms  # type: ignore[import-untyped]
from torchvision.ops.misc import Conv2dNormActivation  # type: ignore[import-untyped]


def get_quantizer():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(tosa_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def get_selective_quantizer_by_module(
    module_types: Dict[torch.nn.Module, QuantizationConfig]
):
    quantizer = get_quantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    for module_type, config in module_types.items():
        quantizer.set_module_type(module_type, config)

    return quantizer


def get_selective_quantizer_by_module_name(module_names: Dict[str, QuantizationConfig]):
    quantizer = get_quantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    for module_name, config in module_names.items():
        quantizer.set_module_name(module_name, config)

    return quantizer


class Add(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class AddSoftmaxAdd(torch.nn.Module):
    module_names = {"add_0": None, "add_1": None}
    module_types = {
        Add: None,
    }
    quantized_aten_targets = {"aten.relu.default": 1}
    non_quantized_aten_targets = {"aten.add.Tensor": 2}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.add_0 = Add()
        self.add_1 = Add()

    def get_inputs(self):
        return (torch.randn(1, 10), torch.randn(1, 10))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.add_0(x, y)
        z = self.relu(z)
        z = self.softmax(z)
        return self.add_1(z, y)


test_models = {
    "add_softmax_add": AddSoftmaxAdd,
}


@common.parametrize("model", test_models)
def test_selective_quant_module_name_tosa_INT(model):
    model = model()
    inputs = model.get_inputs()
    quantzed_aten_targets = model.quantized_aten_targets
    non_quantized_aten_targets = model.non_quantized_aten_targets
    quantization_annotations = {}
    for target, count in quantzed_aten_targets.items():
        quantization_annotations[target] = {
            get_symmetric_quantization_config().output_activation: count
        }
    for target, count in non_quantized_aten_targets.items():
        quantization_annotations[target] = {None: count}

    pipeline = QuantizationPipeline[tuple[torch.Tensor, torch.Tensor]](
        model,
        inputs,
        quantizer=get_selective_quantizer_by_module_name(model.module_names),
        qspecs=quantization_annotations,
    )

    pipeline.run()


@common.parametrize("model", test_models)
def test_selective_quant_module_type_tosa_INT(model):
    model = model()
    inputs = model.get_inputs()
    quantzed_aten_targets = model.quantized_aten_targets
    non_quantized_aten_targets = model.non_quantized_aten_targets
    quantization_annotations = {}
    for target, count in quantzed_aten_targets.items():
        quantization_annotations[target] = {
            get_symmetric_quantization_config().output_activation: count
        }
    for target, count in non_quantized_aten_targets.items():
        quantization_annotations[target] = {None: count}

    pipeline = QuantizationPipeline[tuple[torch.Tensor, torch.Tensor]](
        model,
        inputs,
        quantizer=get_selective_quantizer_by_module(model.module_types),
        qspecs=quantization_annotations,
    )

    pipeline.run()


mv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights)
mv3.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def test_mv3_selective_quant_int16():
    model = mv3
    inputs = (normalize(torch.randn(1, 3, 224, 224)),)

    a16w8_config = get_symmetric_a16w8_quantization_config()
    quantization_annotations = {
        "aten.conv2d.default": {
            a16w8_config.output_activation: 29,
        },
        "aten.hardswish_.default": {
            a16w8_config.output_activation: 18,
        },
        "aten.relu_.default": {
            a16w8_config.output_activation: 5,
        },
    }

    pipeline = QuantizationPipeline[tuple[torch.Tensor]](
        model,
        inputs,
        quantizer=get_selective_quantizer_by_module(
            {
                Conv2dNormActivation: a16w8_config,
            }
        ),
        qspecs=quantization_annotations,
    )

    pipeline.run()


def test_mv3_selective_quant_float32():
    model = mv3
    inputs = (normalize(torch.randn(1, 3, 224, 224)),)

    quantization_annotations = {
        "aten.adaptive_avg_pool2d.default": {
            None: 1,
        },
    }

    pipeline = QuantizationPipeline[tuple[torch.Tensor]](
        model,
        inputs,
        quantizer=get_selective_quantizer_by_module_name(
            {
                "features.11.block.2.avgpool": None,
            }
        ),
        qspecs=quantization_annotations,
    )

    pipeline.run()


def test_mv3_io_quant():
    model = mv3
    inputs = (normalize(torch.randn(1, 3, 224, 224)),)

    quantizer = get_quantizer()
    # Workaround to disable quantization for all modules
    quantizer.set_module_type(torch.nn.Module, None)
    # Only quantize IO
    quantizer.set_io(get_symmetric_quantization_config())

    pipeline = QuantizationPipeline[tuple[torch.Tensor]](
        model,
        inputs,
        quantizer=quantizer,
        input_qspecs={get_symmetric_quantization_config().input_activation: 1},
        output_qspecs={get_symmetric_quantization_config().output_activation: 1},
    )

    pipeline.run()
