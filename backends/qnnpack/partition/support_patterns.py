# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.exir as exir
import torch
import torch.nn.functional as F
from executorch.backends.transforms import apply_addmm_mm_to_linear_transform

from executorch.exir import CaptureConfig

from torch.ao.quantization import QConfig, QConfigMapping  # @manual

from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)

from torch.ao.quantization.observer import (
    default_dynamic_quant_observer,
    default_per_channel_weight_observer,
)

from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)


def get_dynamic_quantized_graph(f, example_inputs, dynamic_shape=False):
    qconfig_mapping = QConfigMapping().set_object_type(
        F.linear,
        QConfig(
            activation=default_dynamic_quant_observer,
            weight=default_per_channel_weight_observer,
        ),
    )

    # Prepare for quantization
    prepared_mod = prepare_fx(
        f,
        qconfig_mapping,
        example_inputs,
        backend_config=get_executorch_backend_config(),
    )

    # Convert module
    converted_mod = _convert_to_reference_decomposed_fx(prepared_mod)
    if dynamic_shape:
        capture_config = CaptureConfig(enable_dynamic_shape=True)
    else:
        capture_config = CaptureConfig()
    # EXIR trace
    gm = (
        exir.capture(converted_mod, example_inputs, capture_config)
        .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
        .exported_program.graph_module
    )

    return apply_addmm_mm_to_linear_transform(gm.graph)


def get_dynamic_quant_addmm_with_view_copy_graph(dynamic_shape=False):
    f = torch.nn.Linear(960, 256).eval()
    example_inputs = (torch.rand(5, 1, 960),)
    return get_dynamic_quantized_graph(f, example_inputs, dynamic_shape)


def get_dynamic_quant_addmm_without_view_copy_graph(dynamic_shape=False):
    example_inputs = (torch.rand(1, 1),)
    f = torch.nn.Linear(1, 1)
    return get_dynamic_quantized_graph(f, example_inputs, dynamic_shape)


def get_dynamic_quant_mm_with_view_copy_graph(dynamic_shape=False):
    example_inputs = (torch.rand(1, 1, 1, 768),)
    f = torch.nn.Linear(768, 4096, bias=False)
    return get_dynamic_quantized_graph(f, example_inputs, dynamic_shape)


def get_dynamic_quant_mm_without_view_copy_graph(dynamic_shape=False):
    example_inputs = (torch.rand(1, 1),)
    f = torch.nn.Linear(1, 1, bias=False)
    return get_dynamic_quantized_graph(f, example_inputs, dynamic_shape)
