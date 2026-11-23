# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable

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
from executorch.exir.pass_manager import PassType
from executorch.export.recipe import ExportRecipe, LoweringRecipe, QuantizationRecipe
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from torch._ops import OpOverload
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


def get_fused_quant_quantization_recipe(
    quantizer: FusedQuantQuantizer,
) -> QuantizationRecipe:
    """The quantize stage of a fused_quant ``ExportRecipe``.

    Decomposes the captured model to core ATen, keeping
    ``quantizer.preserved_ops()`` whole, and calibrates on the example inputs
    given to ``export``, restoring the model's own buffers (a KV cache, say)
    afterwards. Fusion runs after export; see
    :func:`get_fused_quant_aten_transform_passes`.

    Its output is already an exported graph, so the ``ExportRecipe`` that uses
    it sets ``strict=False`` to re-export it without Dynamo.
    """
    buffers = _BufferRestore()
    return QuantizationRecipe(
        quantizers=[quantizer],
        preserve_ops=list(quantizer.preserved_ops()),
        post_prepare_passes=[buffers.save],
        pre_convert_passes=[buffers.restore],
    )


def get_fused_quant_aten_transform_passes(
    quantizer: FusedQuantQuantizer,
) -> list[Callable[[str, ExportedProgram], ExportedProgram]]:
    """The ``aten_transform_passes`` of a fused_quant ``ExportRecipe``: fuse
    what ``quantizer`` quantized."""

    def fuse(method_name: str, program: ExportedProgram) -> ExportedProgram:
        quantizer.fuse(program)
        return program

    return [fuse]


def get_fused_quant_edge_compile_config(
    preserve_ops: Iterable[OpOverload],
) -> EdgeCompileConfig:
    """The ``to_edge`` config of a fused_quant ``ExportRecipe``.

    ``preserve_ops`` are kept whole through ``to_edge``, usually
    ``quantizer.preserved_ops()``. Pass the same ops to
    ``DecomposeFusedQuant(keep_whole=...)``, so that an unclaimed fused op ends up
    as its float counterpart does: the edge verifier accepts the non-core ones it
    re-emits whole, and the others are decomposed. ``_skip_dim_order`` keeps
    classic aten copies instead of dim_order_ops (``_to_dim_order_copy``), which
    not every runtime registers.
    """
    preserved = list(preserve_ops)
    return EdgeCompileConfig(
        _skip_dim_order=True,
        _core_aten_ops_exception_list=preserved,
        preserve_ops=preserved,
    )


@experimental("The fused_quant recipes are experimental and may change without notice.")
def get_fused_quant_portable_recipe(quantizer: FusedQuantQuantizer) -> ExportRecipe:
    """Quantize with ``quantizer`` into a program for the portable kernels.

    The fused_quant pipeline with no backend, through ``to_executorch``. Nothing
    claims anything, so every fused op decomposes back to q/dq plus core ATen.
    """

    def edge_passes(method_name: str, program: ExportedProgram) -> list[PassType]:
        return [
            *get_fused_quant_passes(),
            ToChannelsFirst(),
            DecomposeFusedQuant(),
            AssertNoFusedQuantOps(),
        ]

    return ExportRecipe(
        name="fused_quant_portable",
        strict=False,
        quantization_recipe=get_fused_quant_quantization_recipe(quantizer),
        aten_transform_passes=get_fused_quant_aten_transform_passes(quantizer),
        lowering_recipe=LoweringRecipe(
            edge_transform_passes=[edge_passes],
            edge_compile_config=get_fused_quant_edge_compile_config(()),
        ),
    )
