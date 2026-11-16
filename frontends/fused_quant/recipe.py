# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Optional

import executorch.kernels.quantized  # noqa: F401  registers the q/dq out variants
import torch
from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.exir import EdgeCompileConfig, ExportedProgram
from executorch.exir._warnings import experimental
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_manager import PassType
from executorch.export.recipe import ExportRecipe, LoweringRecipe, QuantizationRecipe
from executorch.export.types import StageType
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from torchao.quantization.pt2e import AffineQuantizedObserverBase, ObserverBase


def _model_buffers(graph_module: torch.fx.GraphModule) -> list[torch.Tensor]:
    """The prepared model's own buffers, without the observers' statistics."""
    observers = tuple(
        f"{name}."
        for name, module in graph_module.named_modules()
        if isinstance(module, (ObserverBase, AffineQuantizedObserverBase))
    )
    return [
        buffer
        for name, buffer in graph_module.named_buffers()
        if not name.startswith(observers)
    ]


class _BufferRestore:
    """Restores the model buffers calibration writes to, such as a KV cache's."""

    def __init__(self) -> None:
        self._saved: list[tuple[torch.Tensor, torch.Tensor]] = []

    def save(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        self._saved = [
            (buffer, buffer.detach().clone()) for buffer in _model_buffers(graph_module)
        ]
        return graph_module

    def restore(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.no_grad():
            for buffer, value in self._saved:
                buffer.copy_(value)
        self._saved = []
        return graph_module


def _fused_quant_recipe(
    quantizer: FusedQuantQuantizer,
    name: str,
    *,
    fused_quant_passes: Callable[[], list[PassType]],
    decompose: bool,
    pipeline_stages: Optional[list[StageType]],
) -> ExportRecipe:
    preserved = list(quantizer.preserved_ops())
    buffers = _BufferRestore()

    def fuse(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        quantizer.fuse(graph_module)
        return graph_module

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        passes: list[PassType] = fused_quant_passes()
        if decompose:
            passes += [ToChannelsFirst(), DecomposeFusedQuant(), AssertNoFusedQuantOps()]
        return passes

    return ExportRecipe(
        name=name,
        # The quantized model is already an exported graph; re-export it without
        # Dynamo.
        strict=False,
        quantization_recipe=QuantizationRecipe(
            quantizers=[quantizer],
            preserve_ops=preserved,
            post_prepare_passes=[buffers.save],
            pre_convert_passes=[buffers.restore],
            post_convert_passes=[fuse],
        ),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            # DecomposeFusedQuant re-emits the ops kept whole for any fused op left
            # unclaimed, and the edge verifier must accept the non-core ones.
            # _skip_dim_order keeps classic aten copies instead of dim_order_ops
            # (_to_dim_order_copy), which not every runtime registers.
            edge_compile_config=EdgeCompileConfig(
                _skip_dim_order=True,
                _core_aten_ops_exception_list=preserved,
                preserve_ops=preserved,
            ),
        ),
        pipeline_stages=pipeline_stages,
    )


@experimental(
    "The fused_quant recipes are experimental and may change without notice."
)
def get_fused_quant_recipe(
    quantizer: FusedQuantQuantizer,
    *,
    absorb_quant_with_fork: bool = False,
    absorb_quant_into_inputs: bool = False,
    symmetric_quant_input_ops: frozenset[EdgeOpOverload] = frozenset(),
    extra_quantizable_ops: dict[EdgeOpOverload, tuple[int, ...]] | None = None,
    channels_last: bool = False,
    fold_into_linear: bool = False,
    prequantize_embedding: bool = False,
    split_linear_at_slices: bool = False,
) -> ExportRecipe:
    """Quantize with ``quantizer`` into the fused_quant edge program a backend lowers.

    The quantize stage decomposes the model to core ATen, keeping
    ``quantizer.preserved_ops()`` whole, and calibrates on the example inputs
    given to ``export``; the quantizer fuses after ``convert_pt2e``. ``to_edge``
    keeps the same ops whole, and the fused_quant passes run on the edge graph,
    configured by the keyword arguments, which are those of
    ``get_fused_quant_passes``. The pipeline stops there:
    ``session.get_edge_program_manager()`` is the program for the backend's
    colorer, lowering and partitioner.
    """
    return _fused_quant_recipe(
        quantizer,
        "fused_quant",
        fused_quant_passes=partial(
            get_fused_quant_passes,
            absorb_quant_with_fork=absorb_quant_with_fork,
            absorb_quant_into_inputs=absorb_quant_into_inputs,
            symmetric_quant_input_ops=symmetric_quant_input_ops,
            extra_quantizable_ops=extra_quantizable_ops,
            channels_last=channels_last,
            fold_into_linear=fold_into_linear,
            prequantize_embedding=prequantize_embedding,
            split_linear_at_slices=split_linear_at_slices,
        ),
        decompose=False,
        pipeline_stages=[
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
        ],
    )


@experimental(
    "The fused_quant recipes are experimental and may change without notice."
)
def get_fused_quant_portable_recipe(quantizer: FusedQuantQuantizer) -> ExportRecipe:
    """Quantize with ``quantizer`` into a program for the portable kernels.

    :func:`get_fused_quant_recipe` with no backend, through ``to_executorch``.
    Nothing claims anything, so every fused op decomposes back to q/dq plus
    ATen. That re-emits the ops the quantizer keeps whole, so they must be core
    ATen, as those of :func:`get_core_aten_quantizers` are.
    """
    non_core = sorted(
        str(op) for op in quantizer.preserved_ops() if torch.Tag.core not in op.tags
    )
    if non_core:
        raise ValueError(
            f"The portable recipe needs a quantizer that keeps only core ATen ops "
            f"whole, but it keeps {non_core}."
        )
    return _fused_quant_recipe(
        quantizer,
        "fused_quant_portable",
        fused_quant_passes=get_fused_quant_passes,
        decompose=True,
        pipeline_stages=None,
    )
