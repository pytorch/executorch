import unittest
from typing import Optional, Tuple

import torch

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.vulkan._passes import FuseQuantizedOpsTransform

from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_linear_weight_only_qcs_xnn_qconfig,
    VulkanQuantizer,
)

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge

from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

###################
## Common Models ##
###################


class SingleLinearModule(torch.nn.Module):
    def __init__(self, K=256, N=128):
        super().__init__()
        self.K = K
        self.N = N
        self.linear = torch.nn.Linear(K, N, bias=False)

    def forward(self, x):
        return self.linear(x)

    def get_sample_inputs(self):
        sample_inputs = (torch.rand(size=(32, self.K), dtype=torch.float32),)
        return sample_inputs


###########
## Tests ##
###########


def quantize_and_lower_module(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    quantizer: Quantizer,
    dynamic_shapes=None,
) -> EdgeProgramManager:
    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # TODO(T182928844): Delegate dim order op to backend.
        _check_ir_validity=False,
    )

    program = torch.export.export_for_training(
        model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()

    program = prepare_pt2e(program, quantizer)  # pyre-ignore
    # Calibrate
    program(*sample_inputs)

    program = convert_pt2e(program)

    program = torch.export.export(program, sample_inputs, dynamic_shapes=dynamic_shapes)

    edge_program = to_edge(
        program,
        compile_config=edge_compile_config,
    )

    return edge_program


def get_target_canonical_name(node: torch.fx.Node) -> Optional[str]:
    if node.op != "call_function":
        return None
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name


def op_node_count(graph_module: torch.fx.GraphModule, canonical_op_name: str) -> int:
    count = 0
    for node in graph_module.graph.nodes:
        canonical_name = get_target_canonical_name(node)
        if canonical_name is not None and canonical_name == canonical_op_name:
            count += 1
    return count


class TestVulkanPasses(unittest.TestCase):

    def test_fuse_int8pack_mm(self):
        K = 256
        N = 256
        model = SingleLinearModule(K, N)
        sample_inputs = model.get_sample_inputs()

        quantizer = VulkanQuantizer()
        quantizer.set_global(get_linear_weight_only_qcs_xnn_qconfig(8))

        edge_manager = quantize_and_lower_module(
            model,
            sample_inputs,
            quantizer,
        )

        ep = edge_manager._edge_programs["forward"]
        edge_manager.transform(
            [
                AddmmToLinearTransform(),
                FuseQuantizedOpsTransform(ep),
            ]
        )

        gm = ep.graph_module

        self.assertEqual(op_node_count(gm, "_weight_int8pack_mm.default"), 1)
        self.assertEqual(op_node_count(gm, "dequantize_per_channel.default"), 0)

    def test_fuse_linear_qcs4w(self):
        K = 256
        N = 256
        model = SingleLinearModule(K, N)
        sample_inputs = model.get_sample_inputs()

        quantizer = VulkanQuantizer()
        quantizer.set_global(get_linear_weight_only_qcs_xnn_qconfig(4))

        edge_manager = quantize_and_lower_module(
            model,
            sample_inputs,
            quantizer,
        )

        ep = edge_manager._edge_programs["forward"]
        edge_manager.transform(
            [
                AddmmToLinearTransform(),
                FuseQuantizedOpsTransform(ep),
            ]
        )

        gm = ep.graph_module

        self.assertEqual(op_node_count(gm, "linear_qcs4w.default"), 1)
        self.assertEqual(op_node_count(gm, "dequantize_per_channel.default"), 0)
