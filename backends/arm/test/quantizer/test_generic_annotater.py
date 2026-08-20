# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools

from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer import is_annotated
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.quantizer.quantization_annotator import annotate_graph
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.test.harness.stages import StageType

from torch.export import export
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


input_t1 = Tuple[torch.Tensor]  # Input x


class SingleOpModel(torch.nn.Module):
    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        example_input: Tuple[Any, ...],
        **op_kwargs: Any,
    ) -> None:
        super().__init__()
        self.op: Callable[..., torch.Tensor] = op
        self._example_input: Tuple[Any, ...] = example_input
        self.op_kwargs: dict[str, Any] = dict(op_kwargs)

    def forward(self, x: Any) -> torch.Tensor:
        return self.op(x, **self.op_kwargs)

    def example_inputs(self) -> Tuple[Any, ...]:
        return self._example_input


def check_annotation(model: SingleOpModel) -> None:
    pipeline = TosaPipelineINT[input_t1](model, model.example_inputs(), [], [])
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    artifact = pipeline.tester.get_artifact(StageType.QUANTIZE)

    partitions = get_source_partitions(artifact.graph, [model.op])
    partitions = list(itertools.chain.from_iterable(partitions.values()))

    assert len(partitions) == 1
    partition = partitions[0]
    assert all(is_annotated(node) for node in partition.nodes)


def test_squeeze_tosa_INT():
    check_annotation(SingleOpModel(torch.squeeze, (torch.rand(8, 8, 1),)))
    check_annotation(SingleOpModel(torch.squeeze_copy, (torch.rand(8, 8, 1),)))


def test_unsqueeze_tosa_INT():
    check_annotation(SingleOpModel(torch.unsqueeze, (torch.rand(8, 8),), dim=0))
    check_annotation(SingleOpModel(torch.unsqueeze_copy, (torch.rand(8, 8),), dim=0))


def test_reshape_tosa_INT():
    check_annotation(
        SingleOpModel(torch.reshape, (torch.randn(8, 8),), shape=(64,)),
    )


def test_view_tosa_INT():
    check_annotation(
        SingleOpModel(torch.view_copy, (torch.randn(4, 4),), size=(2, 8)),
    )


def test_slice_tosa_INT():
    check_annotation(
        SingleOpModel(torch.slice_copy, (torch.randn(3, 4),)),
    )


def test_transpose_tosa_INT():
    check_annotation(
        SingleOpModel(torch.transpose, (torch.randn(2, 3),), dim0=0, dim1=1),
    )
    check_annotation(
        SingleOpModel(torch.transpose_copy, (torch.randn(2, 3),), dim0=0, dim1=1),
    )


def test_moveaxis_movedim_tosa_INT():
    check_annotation(
        SingleOpModel(
            torch.moveaxis,
            (torch.randn(2, 3, 4),),
            source=1,
            destination=-1,
        ),
    )
    check_annotation(
        SingleOpModel(
            torch.moveaxis,
            (torch.randn(2, 3, 4),),
            source=(0, 1),
            destination=(-1, -2),
        ),
    )
    check_annotation(
        SingleOpModel(
            torch.movedim,
            (torch.randn(2, 3, 4),),
            source=1,
            destination=-1,
        ),
    )
    check_annotation(
        SingleOpModel(
            torch.movedim,
            (torch.randn(2, 3, 4),),
            source=(0, 1),
            destination=(-1, -2),
        ),
    )


def test_tile_tosa_INT():
    check_annotation(
        SingleOpModel(torch.tile, (torch.randn(4, 4),), dims=(2,)),
    )


def test_flip_tosa_INT():
    check_annotation(
        SingleOpModel(torch.flip, (torch.randn(2, 4),), dims=(0, 1)),
    )


def test_concat_tosa_INT():
    check_annotation(
        SingleOpModel(
            torch.concatenate, ((torch.randn(2, 3), torch.randn(2, 3)),), dim=0
        ),
    )


class GridSampleModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class GridFloatQuantizationConfig:
    def __init__(self) -> None:
        self.base = get_symmetric_quantization_config()

    def get_input_act_qspec(self, node=None, input_node=None):
        if (
            node is not None
            and input_node is not None
            and node.target == torch.ops.aten.grid_sampler.default
            and input_node == node.args[1]
        ):
            return None
        return self.base.get_input_act_qspec(node, input_node)

    def get_output_act_qspec(self, node=None):
        return self.base.get_output_act_qspec(node)

    def get_weight_qspec(self, node=None):
        return self.base.get_weight_qspec(node)

    def get_bias_qspec(self, node=None):
        return self.base.get_bias_qspec(node)


def test_grid_sampler_annotation_keeps_float_grid_when_grid_qspec_is_none():
    module = GridSampleModule().eval()
    example_inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 2))
    gm = export(module, example_inputs).graph_module

    annotate_graph(gm, GridFloatQuantizationConfig())

    grid_sampler_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.grid_sampler.default
    )
    image_node = grid_sampler_node.args[0]
    grid_node = grid_sampler_node.args[1]
    annotation = grid_sampler_node.meta[Q_ANNOTATION_KEY]

    assert is_annotated(grid_sampler_node)
    assert image_node in annotation.input_qspec_map
    assert grid_node not in annotation.input_qspec_map
    assert annotation.output_qspec is not None


def test_grid_sampler_annotation_keeps_default_tosa_grid_float():
    module = GridSampleModule().eval()
    example_inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 2))
    gm = export(module, example_inputs).graph_module

    annotate_graph(gm, get_symmetric_quantization_config())

    grid_sampler_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.grid_sampler.default
    )
    grid_node = grid_sampler_node.args[1]
    annotation = grid_sampler_node.meta[Q_ANNOTATION_KEY]

    assert grid_node not in annotation.input_qspec_map


def test_vgf_quantizer_quantizes_grid_sampler_grid_coords():
    module = GridSampleModule().eval()
    example_inputs = (torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 2))
    gm = export(module, example_inputs).graph_module

    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.annotate(gm)

    grid_sampler_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.grid_sampler.default
    )
    grid_node = grid_sampler_node.args[1]
    annotation = grid_sampler_node.meta[Q_ANNOTATION_KEY]

    assert grid_node in annotation.input_qspec_map
