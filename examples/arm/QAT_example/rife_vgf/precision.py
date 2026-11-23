# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image ranges and matching host/export precision for the two-block model."""

import copy
from collections import Counter
from dataclasses import replace

import torch
from executorch.backends.arm.quantizer import VgfQuantizer
from torch.fx import GraphModule
from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
TAIL_OPS = {
    torch.ops.aten.grid_sampler.default: 2,
    torch.ops.aten.sigmoid.default: 1,
    torch.ops.aten.sub.Tensor: 1,
    torch.ops.aten.mul.Tensor: 2,
}


class FullRangeImageQuantizer(VgfQuantizer):
    """Use all SNORM-safe codes for [0, 1] images; retain mixed concat ranges."""

    def annotate(self, model):
        model = super().annotate(model)
        image_spec = FixedQParamsQuantizationSpec(
            dtype=torch.int8,
            scale=1 / 254,
            zero_point=-127,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
        )
        images = {n for n in model.graph.nodes if n.op == "placeholder"}
        for node in model.graph.nodes:
            if (
                str(node.target).startswith(
                    (
                        "aten.cat.",
                        "aten.slice.",
                        "aten.alias.",
                        "aten.upsample_bilinear2d.",
                    )
                )
                and node.all_input_nodes
                and all(n in images for n in node.all_input_nodes)
            ):
                images.add(node)
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            if annotation is None:
                continue
            uint8_input = (
                node.op == "placeholder"
                and isinstance(annotation.output_qspec, QuantizationSpec)
                and annotation.output_qspec.dtype == torch.uint8
            )
            if node in images:
                annotation.output_qspec = (
                    replace(
                        image_spec,
                        dtype=torch.uint8,
                        scale=1 / 255,
                        zero_point=0,
                        quant_min=0,
                        quant_max=255,
                    )
                    if uint8_input
                    else image_spec
                )
            if (
                node.target == torch.ops.aten.cat.default
                and node not in images
                and any(source in images for source in annotation.input_qspec_map)
            ):
                sources = list(annotation.input_qspec_map)
                anchor = (sources[0], node)
                annotation.input_qspec_map = {
                    source: (
                        self.global_config.input_activation
                        if i == 0
                        else SharedQuantizationSpec(anchor)
                    )
                    for i, source in enumerate(sources)
                }
                annotation.output_qspec = SharedQuantizationSpec(anchor)
            else:
                for source in annotation.input_qspec_map:
                    if source in images:
                        annotation.input_qspec_map[source] = image_spec
        return model


def floating_tail(model, *, prepared=False):
    """Share weights, bypass final-tail quantization and retain the image output."""
    result = GraphModule(model, copy.deepcopy(model.graph))
    nodes = list(result.graph.nodes)
    (flow,) = [
        n
        for n in nodes
        if n.target == torch.ops.aten.add.Tensor
        and tuple(n.meta["val"].shape[:2]) == (1, 4)
    ]
    samplers = [n for n in nodes if n.target == torch.ops.aten.grid_sampler.default]
    if len(samplers) < 2:
        raise ValueError("Expected two final image warps.")
    start, changed = nodes.index(samplers[-2]), set()
    for index, node in enumerate(nodes):
        if prepared:
            if node.op != "call_module" or not isinstance(
                result.get_submodule(node.target), FakeQuantizeBase
            ):
                continue
            source = node.args[0]
        else:
            if node.target != DQ or node.args[0].target != Q:
                continue
            source = node.args[0].args[0]
        if index > start and source.target in TAIL_OPS:
            node.replace_all_uses_with(source)
            changed.add(source)
        elif source is flow:
            node.replace_all_uses_with(source)
    if Counter(n.target for n in changed) != TAIL_OPS:
        raise ValueError("Expected two warps, sigmoid, complement and two products.")
    result.graph.eliminate_dead_code()
    result.graph.lint()
    result.recompile()
    return result


def _qparams(node):
    if (
        node.target != DQ
        or node.args[0].target != Q
        or node.args[3:6] != (-127, 127, torch.int8)
    ):
        raise ValueError("Expected a signed INT8 per-tensor quantization boundary.")
    return node.args[1:3]


class RifePrecisionModel(torch.nn.Module):
    """Keep a quantized export shadow and its floating-tail host reference."""

    def __init__(self, model):
        super().__init__()
        self.export_model = model
        nodes = list(model.graph.nodes)
        (sigmoid,) = [n for n in nodes if n.target == torch.ops.aten.sigmoid.default]
        (complement,) = [n for n in nodes if n.target == torch.ops.aten.sub.Tensor]
        (flow,) = [
            n
            for n in nodes
            if n.target == Q
            and n.args[0].target == torch.ops.aten.add.Tensor
            and tuple(n.args[0].meta["val"].shape[:2]) == (1, 4)
        ]
        (output,) = model.graph.output_node().args[0]
        if output.target == DQ and output.args[5] == torch.uint8:
            output = output.args[0].args[0]
        constant = complement.args[0]
        one = DQ(model.get_buffer(constant.args[0].target), *constant.args[1:]).item()
        self.policy = {
            "mask": _qparams(sigmoid.args[0]),
            "sigmoid": next(iter(sigmoid.users)).args[1:3],
            "output": _qparams(output),
            "one": one,
            "previous": _qparams(flow.args[0].args[0]),
            "delta": _qparams(flow.args[0].args[1]),
        }
        self.reference = floating_tail(model)

    def forward(self, first, second):
        return self.reference(first, second)
