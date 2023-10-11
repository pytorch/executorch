# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.backends.qnnpack.partition.support_patterns as support_patterns

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
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


def get_actual_dyanmic_quantized_graph(
    graph_module, example_inputs, dynamic_shape=False
):
    qconfig_mapping = QConfigMapping().set_object_type(
        F.linear,
        QConfig(
            activation=default_dynamic_quant_observer,
            weight=default_per_channel_weight_observer,
        ),
    )

    prepared_mod = prepare_fx(
        graph_module,
        qconfig_mapping,
        example_inputs,
        backend_config=get_executorch_backend_config(),
    )

    converted_mod = _convert_to_reference_decomposed_fx(prepared_mod)

    # Step 2: EXIR capturing
    capture_config = CaptureConfig(enable_dynamic_shape=dynamic_shape)
    dynamic_quantized_exir_graph = (
        exir.capture(converted_mod, example_inputs, config=capture_config)
        .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
        .exported_program.graph_module
    )
    dynamic_quantized_exir_graph.graph = apply_addmm_mm_to_linear_transform(
        dynamic_quantized_exir_graph.graph
    )
    return dynamic_quantized_exir_graph.graph


class TestQnnbackends(unittest.TestCase):
    def test_dynamic_quantize_addmm_with_view_copy_partitioner(self):
        example_inputs = (torch.rand(5, 1, 256),)
        in_features = 256
        out_features = 256
        linear_mod = torch.nn.Linear(in_features, out_features).eval()
        for dynamic_shape in [True, False]:
            linear_pattern_graph = (
                support_patterns.get_dynamic_quant_addmm_with_view_copy_graph(
                    dynamic_shape
                )
            )

            actual_dynamic_quant_linear = get_actual_dyanmic_quantized_graph(
                linear_mod, example_inputs, dynamic_shape=dynamic_shape
            )

            subgraph_matcher = SubgraphMatcher(
                linear_pattern_graph, ignore_literals=True
            )
            match_result = subgraph_matcher.match(actual_dynamic_quant_linear)

            self.assertEqual(len(match_result), 1)

    def test_dynamic_quantize_mm_with_view_copy_partitioner(self):
        example_inputs = (torch.rand(1, 1, 1, 768),)
        in_features = 768
        out_features = 4096
        linear_mod = torch.nn.Linear(in_features, out_features, bias=False).eval()
        for dynamic_shape in [True, False]:
            linear_pattern_graph = (
                support_patterns.get_dynamic_quant_mm_with_view_copy_graph(
                    dynamic_shape=dynamic_shape
                )
            )

            actual_dynamic_quant_linear = get_actual_dyanmic_quantized_graph(
                linear_mod, example_inputs, dynamic_shape=dynamic_shape
            )

            subgraph_matcher = SubgraphMatcher(
                linear_pattern_graph, ignore_literals=True
            )
            match_result = subgraph_matcher.match(actual_dynamic_quant_linear)

            self.assertEqual(len(match_result), 1)

    def test_dynamic_quantize_addmm_without_view_copy_partitioner(self):
        example_inputs = (torch.rand(1, 1, 1, 768),)
        in_features = 768
        out_features = 4096
        linear_mod = torch.nn.Linear(in_features, out_features, bias=False).eval()
        for dynamic_shape in [True, False]:
            linear_pattern_graph = (
                support_patterns.get_dynamic_quant_mm_with_view_copy_graph(
                    dynamic_shape=dynamic_shape
                )
            )

            actual_dynamic_quant_linear = get_actual_dyanmic_quantized_graph(
                linear_mod, example_inputs, dynamic_shape=dynamic_shape
            )

            subgraph_matcher = SubgraphMatcher(
                linear_pattern_graph, ignore_literals=True
            )
            match_result = subgraph_matcher.match(actual_dynamic_quant_linear)

            self.assertEqual(len(match_result), 1)
