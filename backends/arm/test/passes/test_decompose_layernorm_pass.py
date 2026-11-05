# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes.decompose_layernorm_pass import (
    DecomposeLayerNormPass,
)
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    QuantizationPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from torchao.quantization.pt2e.quantizer import QuantizationSpec

input_t = Tuple[torch.Tensor]  # Input x


class LayerNorm(torch.nn.Module):
    """Basic layer_norm model using torch.nn.layer_norm layer."""

    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)


def test_decompose_layer_norm_tosa_FP():
    module = LayerNorm()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_native_layer_norm_default": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_edge__ops_aten_view_copy_default",
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_full_default",
            "executorch_exir_dialects_edge__ops_aten_rsqrt_default",
            "executorch_exir_dialects_edge__ops_aten_var_correction",
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
            "executorch_exir_dialects_edge__ops_aten_mean_dim",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 2,
            "executorch_exir_dialects_edge__ops_aten_full_default": 1,
            "executorch_exir_dialects_edge__ops_aten_rsqrt_default": 1,
            "executorch_exir_dialects_edge__ops_aten_var_correction": 1,
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_expand_copy_default"
        ],
        pass_list=[DecomposeLayerNormPass],
    )
    pipeline.run()


def test_layer_norm_decomposition_preserves_module_metadata():
    """The normalization mean and subtraction retain their originating module.

    Both generated nodes used to omit ``from_node=node``, which dropped their
    ``nn_module_stack`` and made selective quantization unreliable.

    """
    model = LayerNorm().eval()
    graph_module = torch.export.export(model, model.get_inputs()).module()
    source_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.layer_norm.default
    ]
    assert len(source_nodes) == 1
    original_stack = source_nodes[0].meta.get("nn_module_stack")
    assert original_stack, "Export must retain LayerNorm module provenance"

    result = DecomposeLayerNormPass(tfa_pass=True)(graph_module)
    nodes = list(result.graph_module.graph.nodes)
    for op in (torch.ops.aten.mean.dim, torch.ops.aten.sub.Tensor):
        generated = [node for node in nodes if node.target == op]
        assert len(generated) == 1, f"Expected exactly one generated {op}"
        assert generated[0].meta.get("nn_module_stack") == original_stack


@pytest.mark.parametrize("use_composable_quantizer", [False, True])
def test_layer_norm_decomposition_a16w8_module_override_metadata(
    use_composable_quantizer: bool,
):
    model = LayerNorm().eval()

    a16w8_config = get_symmetric_a16w8_quantization_config()
    a16w8_qspec = a16w8_config.output_activation
    assert isinstance(a16w8_qspec, QuantizationSpec)

    quantizer = TOSAQuantizer(
        TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
        use_composable_quantizer=use_composable_quantizer,
    )
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_module_type(torch.nn.LayerNorm, a16w8_config)

    pipeline = QuantizationPipeline[input_t](
        model,
        model.get_inputs(),
        quantizer=quantizer,
        qspecs={
            "aten.sub.Tensor": {
                a16w8_qspec: 2,
            },
        },
    )
    pipeline.run()
