# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    get_uint8_io_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class CloneAtIoBoundary(torch.nn.Module):
    """Zero-arithmetic cluster whose only adjacent annotated neighbours are
    uint8-annotated IO nodes (input placeholder + graph output).

    With set_global(int8) + set_io(uint8), both the placeholder and the output
    node carry uint8 qspecs that _skip_shared_qspec_from_io filters out, leaving
    adjacent_qspecs empty. Before the IO-boundary fallback fix in
    SharedQspecQuantizer, this caused the cluster to stay in float.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clone(x)


class CatWithHighRangeBranch(torch.nn.Module):
    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        image = torch.cat([img0, img1], dim=1)
        high_range = image * 20.0
        merged = torch.cat([image, high_range], dim=1)
        return torch.clone(merged)


def _get_observer_scale(prepared, observer_node_name: str) -> float:
    observer = prepared.get_submodule(observer_node_name)
    scale, _ = observer.calculate_qparams()
    return float(scale)


def test_uint8_io_quantization_config_tosa_INT_applies_to_io():
    model = SimpleMLP().eval()
    test_data = (torch.rand(1, 4),)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_io(get_uint8_io_quantization_config())

    io_config = get_uint8_io_quantization_config()
    pipeline = QuantizationPipeline(
        model,
        test_data,
        quantizer=quantizer,
        input_qspecs={io_config.input_activation: 1},
        output_qspecs={io_config.output_activation: 1},
    )
    pipeline.run()


def test_io_boundary_shared_cluster_is_quantized():
    """Regression: a zero-arithmetic cluster adjacent only to uint8-annotated IO
    nodes must be annotated with the global int8 qspec, not left in float.

    _skip_shared_qspec_from_io filters the uint8 qspec from IO nodes, so when
    the cluster's only neighbours are such nodes adjacent_qspecs ends up empty.
    The fix in SharedQspecQuantizer detects the IO-boundary via
    _is_quantized_io_boundary and falls back to global_config.get_input_act_qspec().
    """
    model = CloneAtIoBoundary().eval()
    test_data = (torch.rand(1, 4),)
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")

    quantizer = TOSAQuantizer(compile_spec, use_composable_quantizer=True)
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_io(get_uint8_io_quantization_config())

    exported = torch.export.export(model, test_data, strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)

    clone_nodes = [
        n
        for n in prepared.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    assert len(clone_nodes) == 1, f"Expected 1 clone node, got {len(clone_nodes)}"
    clone_node = clone_nodes[0]

    assert (
        Q_ANNOTATION_KEY in clone_node.meta
    ), "clone node was not annotated — IO-boundary cluster stayed in float"
    assert (
        clone_node.meta[Q_ANNOTATION_KEY].output_qspec is not None
    ), "clone node has no output_qspec — IO-boundary cluster stayed in float"


def test_cat_does_not_bridge_shared_qspec_clusters():
    """Regression: cat must not merge image IO and high-range activations into
    one fallback shared-qspec observer clique.
    """
    model = CatWithHighRangeBranch().eval()
    test_data = (torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+INT")

    tosa_quantizer = TOSAQuantizer(compile_spec, use_composable_quantizer=True)
    tosa_quantizer.set_global(get_symmetric_quantization_config())
    tosa_quantizer.set_io(get_uint8_io_quantization_config())

    exported = torch.export.export(model, test_data, strict=True)
    prepared = prepare_pt2e(exported.module(), tosa_quantizer)
    prepared(*test_data)

    graph_nodes = {node.name: node for node in prepared.graph.nodes}
    img0_observer = next(iter(graph_nodes["img0"].users))
    img1_observer = next(iter(graph_nodes["img1"].users))
    final_cat_observer = next(iter(graph_nodes["cat_1"].users))

    assert _get_observer_scale(prepared, img0_observer.target) < 0.01
    assert _get_observer_scale(prepared, img1_observer.target) < 0.01
    assert _get_observer_scale(prepared, final_cat_observer.target) > 0.05
