# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes.decompose_groupnorm_pass import (
    DecomposeGroupNormPass,
)
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from executorch.backends.arm.tosa import TosaSpecification
from torchao.quantization.pt2e.quantizer import QuantizationSpec

input_t = Tuple[torch.Tensor]


class GroupNorm(torch.nn.Module):
    """A module instance whose FX provenance must survive decomposition."""

    def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=2, num_channels=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 4, 4, 4),)


def test_group_norm_decomposition_preserves_module_metadata():
    """Decomposed mean/sub must retain the original GroupNorm module stack."""
    model = GroupNorm().eval()
    graph_module = torch.export.export(model, model.get_inputs()).module()
    source_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.group_norm.default
    ]
    assert len(source_nodes) == 1
    original_stack = source_nodes[0].meta.get("nn_module_stack")
    assert original_stack, "Export must retain GroupNorm module provenance"

    result = DecomposeGroupNormPass(tfa_pass=True)(graph_module)
    nodes = list(result.graph_module.graph.nodes)
    for op in (torch.ops.aten.mean.dim, torch.ops.aten.sub.Tensor):
        generated = [node for node in nodes if node.target == op]
        assert len(generated) == 1, f"Expected exactly one generated {op}"
        assert generated[0].meta.get("nn_module_stack") == original_stack


@pytest.mark.parametrize("use_composable_quantizer", [False, True])
def test_group_norm_decomposition_a16w8_module_override_metadata(
    use_composable_quantizer: bool,
):
    """The GroupNorm-specific A16W8 override must beat the global A8W8
    config.
    """
    model = GroupNorm().eval()

    a16w8_config = get_symmetric_a16w8_quantization_config()
    a16w8_qspec = a16w8_config.output_activation
    assert isinstance(a16w8_qspec, QuantizationSpec)

    quantizer = TOSAQuantizer(
        TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
        use_composable_quantizer=use_composable_quantizer,
    )
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_module_type(torch.nn.GroupNorm, a16w8_config)

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
