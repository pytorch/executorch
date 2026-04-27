# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import executorch.exir

import torch
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.arm_quantizer_utils import SharedQspecQuantizer
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@dataclass(frozen=True)
class OpCase:
    target: Callable
    module: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    expected_output_dtype: torch.dtype | None


def _build_module(
    op_call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.nn.Module:
    class _Module(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = op_call(x, y)
            return out

    return _Module()


def _quantize_and_export(
    model: torch.nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor]
) -> torch.nn.Module:

    exported_model = export(model.eval(), example_inputs, strict=True).module()
    quantizer = CortexMQuantizer()
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(*example_inputs)
    quantized_model = convert_pt2e(prepared_model)
    exported = export(quantized_model, example_inputs, strict=True)
    edge_program = executorch.exir.to_edge(
        exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    fold_pass = FoldAndAnnotateQParamsPass(edge_program.exported_program())
    edge_program = edge_program.transform([fold_pass])
    return edge_program.exported_program().module()


OP_CASES = {
    "clone": OpCase(
        torch.ops.aten.clone.default,
        _build_module(lambda x, y: torch.ops.aten.clone.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "lift_fresh_copy": OpCase(
        torch.ops.aten.lift_fresh_copy.default,
        _build_module(lambda x, y: torch.ops.aten.lift_fresh_copy.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "detach_": OpCase(
        torch.ops.aten.detach_.default,
        _build_module(lambda x, y: torch.ops.aten.detach_.default(x.clone())),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "alias": OpCase(
        torch.ops.aten.alias.default,
        _build_module(lambda x, y: torch.ops.aten.alias.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "alias_copy": OpCase(
        torch.ops.aten.alias_copy.default,
        _build_module(lambda x, y: torch.ops.aten.alias_copy.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "copy_": OpCase(
        torch.ops.aten.copy_.default,
        _build_module(lambda x, y: torch.ops.aten.copy_.default(x.clone(), y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "detach_copy": OpCase(
        torch.ops.aten.detach_copy.default,
        _build_module(lambda x, y: torch.ops.aten.detach_copy.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "unfold_copy": OpCase(
        torch.ops.aten.unfold_copy.default,
        _build_module(lambda x, y: torch.ops.aten.unfold_copy.default(x, 1, 2, 1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "unbind": OpCase(
        torch.ops.aten.unbind.int,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.unbind.int(x, 0), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "minimum": OpCase(
        torch.ops.aten.minimum.default,
        _build_module(lambda x, y: torch.minimum(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "maximum": OpCase(
        torch.ops.aten.maximum.default,
        _build_module(lambda x, y: torch.maximum(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "min_dim": OpCase(
        torch.ops.aten.min.dim,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.min.dim(x, 1, False), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "max_dim": OpCase(
        torch.ops.aten.max.dim,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.max.dim(x, 1, False), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "amin": OpCase(
        torch.ops.aten.amin.default,
        _build_module(lambda x, y: torch.ops.aten.amin.default(x, [1], False)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "amax": OpCase(
        torch.ops.aten.amax.default,
        _build_module(lambda x, y: torch.ops.aten.amax.default(x, [1], False)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "permute": OpCase(
        torch.ops.aten.permute.default,
        _build_module(lambda x, y: torch.ops.aten.permute.default(x, [0, 2, 3, 1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "permute_copy": OpCase(
        torch.ops.aten.permute_copy.default,
        _build_module(
            lambda x, y: torch.ops.aten.permute_copy.default(x, [0, 2, 3, 1])
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "transpose_int": OpCase(
        torch.ops.aten.transpose.int,
        _build_module(lambda x, y: torch.ops.aten.transpose.int(x, 2, 3)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "transpose_copy": OpCase(
        torch.ops.aten.transpose_copy.int,
        _build_module(lambda x, y: torch.ops.aten.transpose_copy.int(x, 2, 3)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "t_copy": OpCase(
        torch.ops.aten.t_copy.default,
        _build_module(lambda x, y: torch.ops.aten.t_copy.default(x)),
        (torch.randn(3, 4), torch.randn(3, 4)),
        None,
    ),
    "t": OpCase(
        torch.ops.aten.t.default,
        _build_module(lambda x, y: torch.ops.aten.t.default(x)),
        (torch.randn(3, 4), torch.randn(3, 4)),
        None,
    ),
    "repeat": OpCase(
        torch.ops.aten.repeat.default,
        _build_module(lambda x, y: torch.ops.aten.repeat.default(x, [1, 2, 1, 1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "repeat_interleave": OpCase(
        torch.ops.aten.repeat_interleave.self_int,
        _build_module(lambda x, y: torch.ops.aten.repeat_interleave.self_int(x, 2, 0)),
        (torch.randn(6), torch.randn(6)),
        None,
    ),
    "expand_copy": OpCase(
        torch.ops.aten.expand_copy.default,
        _build_module(lambda x, y: torch.ops.aten.expand_copy.default(x, [2, 3, 4, 5])),
        (torch.randn(1, 3, 4, 5), torch.randn(1, 3, 4, 5)),
        None,
    ),
    "expand": OpCase(
        torch.ops.aten.expand.default,
        _build_module(lambda x, y: torch.ops.aten.expand.default(x, [2, 3, 4, 5])),
        (torch.randn(1, 3, 4, 5), torch.randn(1, 3, 4, 5)),
        None,
    ),
    "select": OpCase(
        torch.ops.aten.select.int,
        _build_module(lambda x, y: torch.ops.aten.select.int(x, 1, 0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "select_copy": OpCase(
        torch.ops.aten.select_copy.int,
        _build_module(lambda x, y: torch.ops.aten.select_copy.int(x, 1, 0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "slice": OpCase(
        torch.ops.aten.slice.Tensor,
        _build_module(lambda x, y: torch.ops.aten.slice.Tensor(x, 2, 1, 3, 1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "slice_copy": OpCase(
        torch.ops.aten.slice_copy.Tensor,
        _build_module(lambda x, y: torch.ops.aten.slice_copy.Tensor(x, 2, 1, 3, 1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "split": OpCase(
        torch.ops.aten.split.Tensor,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.split.Tensor(x, 2, 1), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "split_with_sizes": OpCase(
        torch.ops.aten.split_with_sizes.default,
        _build_module(
            lambda x, y: operator.getitem(
                torch.ops.aten.split_with_sizes.default(x, [1, 2], 1), 0
            )
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "split_copy": OpCase(
        torch.ops.aten.split_copy.Tensor,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.split_copy.Tensor(x, 2, 1), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "tile": OpCase(
        torch.ops.aten.tile.default,
        _build_module(lambda x, y: torch.ops.aten.tile.default(x, [1, 2, 1, 1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "flip": OpCase(
        torch.ops.aten.flip.default,
        _build_module(lambda x, y: torch.ops.aten.flip.default(x, [2])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "index_select": OpCase(
        torch.ops.aten.index_select.default,
        _build_module(
            lambda x, y: torch.ops.aten.index_select.default(x, 1, torch.tensor([0, 2]))
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(1)),
        torch.int64,
    ),
    "index_put": OpCase(
        torch.ops.aten.index_put.default,
        _build_module(
            lambda x, y: torch.ops.aten.index_put.default(
                x, (torch.tensor([1, 3]),), torch.tensor([1.0, 2.0]), False
            )
        ),
        (torch.randn(6), torch.randn(6)),
        torch.int64,
    ),
    "contiguous": OpCase(
        torch.ops.aten.contiguous.default,
        _build_module(lambda x, y: torch.ops.aten.contiguous.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "as_strided_copy": OpCase(
        torch.ops.aten.as_strided_copy.default,
        _build_module(
            lambda x, y: torch.ops.aten.as_strided_copy.default(
                x, (2, 2), x.stride(), 0
            )
        ),
        (torch.randn(4, 4), torch.randn(4, 4)),
        None,
    ),
    "pixel_shuffle": OpCase(
        torch.ops.aten.pixel_shuffle.default,
        _build_module(lambda x, y: torch.ops.aten.pixel_shuffle.default(x, 2)),
        (torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)),
        None,
    ),
    "pixel_unshuffle": OpCase(
        torch.ops.aten.pixel_unshuffle.default,
        _build_module(lambda x, y: torch.ops.aten.pixel_unshuffle.default(x, 2)),
        (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4)),
        None,
    ),
    "cat": OpCase(
        torch.ops.aten.cat.default,
        _build_module(lambda x, y: torch.ops.aten.cat.default([x, y], 1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "concatenate": OpCase(
        torch.ops.aten.concatenate.default,
        _build_module(lambda x, y: torch.ops.aten.concatenate.default([x, y], 1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "stack": OpCase(
        torch.ops.aten.stack.default,
        _build_module(lambda x, y: torch.ops.aten.stack.default([x, y], 0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "dropout": OpCase(
        torch.ops.aten.dropout.default,
        _build_module(lambda x, y: torch.ops.aten.dropout.default(x, 0.1, False)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "dropout_": OpCase(
        torch.ops.aten.dropout_.default,
        _build_module(
            lambda x, y: torch.ops.aten.dropout_.default(x.clone(), 0.1, False)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "chunk": OpCase(
        torch.ops.aten.chunk.default,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.chunk.default(x, 2, 1), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "index_tensor": OpCase(
        torch.ops.aten.index.Tensor,
        _build_module(
            lambda x, y: torch.ops.aten.index.Tensor(x, [torch.tensor([0, 2])])
        ),
        (torch.randn(4, 4), torch.randn(4, 4)),
        torch.int64,
    ),
    "gather": OpCase(
        torch.ops.aten.gather.default,
        _build_module(
            lambda x, y: torch.ops.aten.gather.default(
                x, 1, torch.tensor([[0, 1, 2], [1, 2, 0]])
            )
        ),
        (torch.randn(2, 3), torch.randn(2, 3)),
        torch.int64,
    ),
    "getitem": OpCase(
        operator.getitem,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.split.Tensor(x, 2, 1), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "squeeze": OpCase(
        torch.ops.aten.squeeze.default,
        _build_module(lambda x, y: torch.ops.aten.squeeze.default(x)),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "squeeze_copy": OpCase(
        torch.ops.aten.squeeze_copy.default,
        _build_module(lambda x, y: torch.ops.aten.squeeze_copy.default(x)),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "squeeze_copy_dim": OpCase(
        torch.ops.aten.squeeze_copy.dim,
        _build_module(lambda x, y: torch.ops.aten.squeeze_copy.dim(x, 0)),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "squeeze_dim": OpCase(
        torch.ops.aten.squeeze.dim,
        _build_module(lambda x, y: torch.ops.aten.squeeze.dim(x, 0)),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "squeeze_dims": OpCase(
        torch.ops.aten.squeeze.dims,
        _build_module(lambda x, y: torch.ops.aten.squeeze.dims(x, [0, 2])),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "squeeze__dim": OpCase(
        torch.ops.aten.squeeze_.dim,
        _build_module(lambda x, y: torch.ops.aten.squeeze_.dim(x.clone(), 0)),
        (torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4)),
        None,
    ),
    "unsqueeze": OpCase(
        torch.ops.aten.unsqueeze.default,
        _build_module(lambda x, y: torch.ops.aten.unsqueeze.default(x, 0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "unsqueeze_copy": OpCase(
        torch.ops.aten.unsqueeze_copy.default,
        _build_module(lambda x, y: torch.ops.aten.unsqueeze_copy.default(x, 0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "reshape": OpCase(
        torch.ops.aten.reshape.default,
        _build_module(lambda x, y: torch.ops.aten.reshape.default(x, [2, -1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "view": OpCase(
        torch.ops.aten.view.default,
        _build_module(lambda x, y: torch.ops.aten.view.default(x, [2, -1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "view_as": OpCase(
        torch.ops.aten.view_as.default,
        _build_module(lambda x, y: torch.ops.aten.view_as.default(x, y)),
        (torch.randn(2, 3, 4), torch.randn(2, 12)),
        None,
    ),
    "view_copy": OpCase(
        torch.ops.aten.view_copy.default,
        _build_module(lambda x, y: torch.ops.aten.view_copy.default(x, [2, -1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "unsafe_view": OpCase(
        torch.ops.aten._unsafe_view.default,
        _build_module(lambda x, y: torch.ops.aten._unsafe_view.default(x, [2, -1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "unflatten": OpCase(
        torch.ops.aten.unflatten.int,
        _build_module(lambda x, y: torch.ops.aten.unflatten.int(x, 1, (2, 3))),
        (torch.randn(2, 6), torch.randn(2, 6)),
        None,
    ),
    "flatten": OpCase(
        torch.ops.aten.flatten.using_ints,
        _build_module(lambda x, y: torch.ops.aten.flatten.using_ints(x, 1, -1)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "pad": OpCase(
        torch.ops.aten.pad.default,
        _build_module(lambda x, y: torch.ops.aten.pad.default(x, [1, 1, 1, 1])),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "constant_pad_nd": OpCase(
        torch.ops.aten.constant_pad_nd.default,
        _build_module(
            lambda x, y: torch.ops.aten.constant_pad_nd.default(x, [1, 1, 1, 1], 0.5)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "clamp": OpCase(
        torch.ops.aten.clamp.default,
        _build_module(lambda x, y: torch.ops.aten.clamp.default(x, -1.0, 1.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "clamp_tensor": OpCase(
        torch.ops.aten.clamp.Tensor,
        _build_module(
            lambda x, y: torch.ops.aten.clamp.Tensor(
                x, torch.full_like(x, -1.0), torch.full_like(x, 1.0)
            )
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "hardtanh": OpCase(
        torch.ops.aten.hardtanh.default,
        _build_module(lambda x, y: torch.ops.aten.hardtanh.default(x, -1.0, 1.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "hardtanh_": OpCase(
        torch.ops.aten.hardtanh_.default,
        _build_module(
            lambda x, y: torch.ops.aten.hardtanh_.default(x.clone(), -1.0, 1.0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "relu": OpCase(
        torch.ops.aten.relu.default,
        _build_module(lambda x, y: torch.ops.aten.relu.default(x)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "relu_": OpCase(
        torch.ops.aten.relu_.default,
        _build_module(lambda x, y: torch.ops.aten.relu_.default(x.clone())),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "eq_tensor": OpCase(
        torch.ops.aten.eq.Tensor,
        _build_module(lambda x, y: torch.ops.aten.eq.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "eq_scalar": OpCase(
        torch.ops.aten.eq.Scalar,
        _build_module(lambda x, y: torch.ops.aten.eq.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "ne_tensor": OpCase(
        torch.ops.aten.ne.Tensor,
        _build_module(lambda x, y: torch.ops.aten.ne.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "ne_scalar": OpCase(
        torch.ops.aten.ne.Scalar,
        _build_module(lambda x, y: torch.ops.aten.ne.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "ge_tensor": OpCase(
        torch.ops.aten.ge.Tensor,
        _build_module(lambda x, y: torch.ops.aten.ge.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "ge_scalar": OpCase(
        torch.ops.aten.ge.Scalar,
        _build_module(lambda x, y: torch.ops.aten.ge.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "gt_tensor": OpCase(
        torch.ops.aten.gt.Tensor,
        _build_module(lambda x, y: torch.ops.aten.gt.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "gt_scalar": OpCase(
        torch.ops.aten.gt.Scalar,
        _build_module(lambda x, y: torch.ops.aten.gt.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "le_tensor": OpCase(
        torch.ops.aten.le.Tensor,
        _build_module(lambda x, y: torch.ops.aten.le.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "le_scalar": OpCase(
        torch.ops.aten.le.Scalar,
        _build_module(lambda x, y: torch.ops.aten.le.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "lt_tensor": OpCase(
        torch.ops.aten.lt.Tensor,
        _build_module(lambda x, y: torch.ops.aten.lt.Tensor(x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "lt_scalar": OpCase(
        torch.ops.aten.lt.Scalar,
        _build_module(lambda x, y: torch.ops.aten.lt.Scalar(x, 0.0)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        torch.bool,
    ),
    "where_self": OpCase(
        torch.ops.aten.where.self,
        _build_module(lambda x, y: torch.ops.aten.where.self(x > 0, x, y)),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "where_default": OpCase(
        torch.ops.aten.where.default,
        _build_module(
            lambda x, y: operator.getitem(torch.ops.aten.where.default(x), 0)
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "while_loop": OpCase(
        torch.ops.higher_order.while_loop,
        _build_module(
            lambda x, y: operator.getitem(
                torch.ops.higher_order.while_loop(
                    lambda value: torch.gt(
                        value.sum(), torch.full((1,), 0.0)
                    ).squeeze(),
                    lambda value: (value - torch.full((1,), 1.0),),
                    (x,),
                    (),
                ),
                0,
            ),
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
    "cond": OpCase(
        torch.ops.higher_order.cond,
        _build_module(
            lambda x, y: torch.cond(
                x.sum() > 0,
                lambda t: t + torch.full((1,), 1.0),
                lambda t: t - torch.full((1,), 1.0),
                (x,),
            ),
        ),
        (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)),
        None,
    ),
}

xfails: dict[str, xfail_type] = {
    "contiguous": "MLETORCH-1863: Contiguos no-op is removed in to-edge, leading to unnecessary Q-DQ-Q-DQ chain.",
    "eq_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "ne_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "ge_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "gt_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "le_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "lt_scalar": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "where_self": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "where_default": "MLETORCH-1865: Properly support flaky scalar comparison ops.",
    "while_loop": "MLETORCH-1866: Support higher-order operators",
    "cond": "MLETORCH-1866: Support higher-order operators",
}


@parametrize(
    "op_case",
    OP_CASES,
    xfails=xfails,
    strict=False,
    skips={"while_loop": "Has been observed to hang randomly."},
)
def test_shared_qspec_portable_int8_ops(op_case: OpCase) -> None:
    tester = CortexMTester(op_case.module, op_case.example_inputs)
    tester.test_dialect(ops_before_transforms={}, ops_after_transforms={})

    module = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    target_nodes = (
        node
        for node in module.graph.nodes
        if node.op == "call_function"
        and not node.target == exir_ops.edge.cortex_m.quantize_per_tensor.default
    )
    target_node = next(target_nodes)
    output_dtype = get_first_fake_tensor(target_node).dtype

    if op_case.expected_output_dtype is None:
        # Regular case, int8 output is expected.
        assert (
            output_dtype == torch.int8
        ), f"{target_node.name} output dtype {output_dtype}"
    elif op_case.expected_output_dtype == torch.int64:
        # Indexing ops like `aten.index_select` and `aten.gather` decompose into two ops, one producing int64 output and the other producing int8 output.
        assert (
            output_dtype == torch.int64
        ), f"{target_node.name} output dtype {output_dtype}"
        target_node = next(target_nodes)
        output_dtype = get_first_fake_tensor(target_node).dtype
        assert (
            output_dtype == torch.int8
        ), f"{target_node.name} output dtype {output_dtype}"
    else:
        # Ops like `aten.eq` and `aten.gt` produce bool output, which is not quantized, so we expect the original output dtype to be preserved. Instead check input.
        assert (
            output_dtype == op_case.expected_output_dtype
        ), f"{target_node.name} output dtype {output_dtype}"
        for input_node in target_node.all_input_nodes:
            fake_tensor = get_first_fake_tensor(input_node)
            assert (
                fake_tensor.dtype == torch.int8
            ), f"{target_node.name} input dtype {output_dtype}"


FVP_OP_CASES_LIST = [
    "clone",
    "lift_fresh_copy",
    "detach_",
    "amax",
    "select",
    "select_copy",
    "cat",
    "concatenate",
    "stack",
    "unsqueeze",
    "reshape",
    "view",
    "flatten",
    "ge_tensor",
]
FVP_OP_CASES = {key: val for key, val in OP_CASES.items() if key in FVP_OP_CASES_LIST}


@parametrize("op_case", FVP_OP_CASES)
def test_shared_qspec_portable_int8_ops_fvp(op_case: OpCase) -> None:
    tester = CortexMTester(op_case.module, op_case.example_inputs)
    tester.test_implementation()


def test_shared_qspec_ops_default_covered() -> None:
    expected = set(SharedQspecQuantizer.SHARED_QSPEC_OPS_DEFAULT)
    covered = {case.target for case in OP_CASES.values()}
    assert expected == covered
