# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, ClassVar, Dict, Tuple

import pytest
import torch
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.exir.dialects._ops import ops as exir_ops


input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, y

_MIXED_PROFILE_PARTIAL_BINARY_TENSOR_TARGETS = (
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.eq.Tensor,
    exir_ops.edge.aten.ge.Tensor,
    exir_ops.edge.aten.gt.Tensor,
    exir_ops.edge.aten.le.Tensor,
    exir_ops.edge.aten.lt.Tensor,
)


class SimpleQuantizeModel(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {
        "rand": (torch.rand(1, 1280, 7, 7), torch.rand(1, 1280, 7, 7)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + torch.maximum((x + x), (y + y))


@common.parametrize("test_data", SimpleQuantizeModel.test_data)
def test_fold_and_annotate_q_params_tosa_INT(test_data: input_t) -> None:
    """Tests the FoldAndAnnotateQParamsPass which folds dq/q nodes into the node
    and stores the quantization parameters in meta.

    Check that the pass runs for add operation and that one q node and one dq
    node is removed from the representation.

    """
    module = SimpleQuantizeModel()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 7,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 6,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        },
        pass_list=[FoldAndAnnotateQParamsPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()


@pytest.mark.parametrize(
    "binary_target",
    (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.sub.Tensor),
)
def test_fold_qdq_preserves_default_partial_binary_qdq(
    binary_target: Callable[..., object],
) -> None:
    _check_fold_qdq_preserves_partial_binary_qdq(binary_target)


@pytest.mark.parametrize(
    "binary_target",
    _MIXED_PROFILE_PARTIAL_BINARY_TENSOR_TARGETS,
)
def test_fold_qdq_preserves_mixed_profile_partial_binary_tensor_qdq(
    binary_target: Callable[..., object],
) -> None:
    _check_fold_qdq_preserves_partial_binary_qdq(
        binary_target, preserve_partial_binary_tensor_qdq=True
    )


def test_fold_qdq_preserves_mixed_profile_partial_grid_sampler_qdq() -> None:
    _check_fold_qdq_preserves_partial_binary_qdq(
        exir_ops.edge.aten.grid_sampler_2d.default,
        preserve_partial_binary_tensor_qdq=True,
        extra_args=(0, 0, False),
    )


def test_fold_qdq_mixed_profile_allowlist_has_test_coverage() -> None:
    tested_targets = {
        *_MIXED_PROFILE_PARTIAL_BINARY_TENSOR_TARGETS,
        exir_ops.edge.aten.grid_sampler_2d.default,
    }

    assert tested_targets == set(
        FoldAndAnnotateQParamsPass._mixed_profile_partial_binary_qdq_targets  # noqa: SLF001
    )


def test_fold_qdq_folds_default_partial_mul_qdq() -> None:
    _, mul, _, _ = _partial_binary_qdq_graph(exir_ops.edge.aten.mul.Tensor)

    assert not FoldAndAnnotateQParamsPass()._has_partial_binary_tensor_qdq_inputs(  # noqa: SLF001
        mul, {0: object()}  # type: ignore[dict-item]
    )


@pytest.mark.parametrize(
    "target",
    (
        exir_ops.edge.aten.index_select.default,
        exir_ops.edge.aten.gather.default,
    ),
)
def test_fold_qdq_does_not_treat_index_as_binary_operand(
    target: Callable[..., object],
) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    index = graph.placeholder("index")
    node = graph.call_function(target, (x, 0, index))

    assert not FoldAndAnnotateQParamsPass(
        preserve_partial_binary_tensor_qdq=True
    )._has_partial_binary_tensor_qdq_inputs(  # noqa: SLF001
        node, {0: object()}  # type: ignore[dict-item]
    )


def _check_fold_qdq_preserves_partial_binary_qdq(
    binary_target: Callable[..., object],
    preserve_partial_binary_tensor_qdq: bool = False,
    extra_args: tuple[int | bool, ...] = (),
) -> None:
    graph_module, add, _, y = _partial_binary_qdq_graph(binary_target, extra_args)
    x_dq = add.args[0]
    add_q = next(iter(add.users))

    FoldAndAnnotateQParamsPass(
        preserve_partial_binary_tensor_qdq=preserve_partial_binary_tensor_qdq
    )(graph_module)

    assert set(add.meta["input_qparams"]) == {0}
    assert set(add.meta["output_qparams"]) == {0}
    assert add.args == (x_dq, y, *extra_args)
    assert add_q in add.users


def _partial_binary_qdq_graph(
    binary_target: Callable[..., object],
    extra_args: tuple[int | bool, ...] = (),
) -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x_q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (x, 0.5, 0, -128, 127, torch.int8),
    )
    x_dq = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        (x_q, 0.5, 0, -128, 127, torch.int8),
    )
    add = graph.call_function(binary_target, (x_dq, y, *extra_args))
    add.meta["custom"] = {
        ArmAnnotationInfo.CUSTOM_META_KEY: ArmAnnotationInfo(quantized=True)
    }
    add_q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (add, 0.5, 0, -128, 127, torch.int8),
    )
    out = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        (add_q, 0.5, 0, -128, 127, torch.int8),
    )
    graph.output(out)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    return graph_module, add, x_q, y
