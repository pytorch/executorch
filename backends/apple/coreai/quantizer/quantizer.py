# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core AI quantization front-end for ExecuTorch.

Keeps the canonical ExecuTorch quantization pipeline::

    prepare -> calibrate -> convert -> export -> to_edge_transform_and_lower

``CoreAIQuantizer.convert()`` produces the export-ready, fully quantized graph
(Core AI ``coreai::`` ops). It intentionally runs the entire ``coreai_opt``
``finalize(CoreAI)`` (``convert_pt2e`` + conv/bn fold + Q/DQ to ``coreai::``
rewrite + kv-cache relocation), because that whole rewrite must run before
``torch.export``: a graph still carrying ``coreai_opt`` fake-quant cannot be
strict-``torch.export``ed (its ``FakeQuantize.forward`` has a data-dependent
guard). Nothing quant-related is left for the backend ``preprocess`` to do; it
simply lowers the ``coreai::`` ops via ``add_exported_program``.

``coreai_opt`` (the ``coreai-optimization`` package) is imported lazily so this
module stays importable in environments without it installed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from torch import fx, nn


class CoreAIQuantizer:
    """PT2E quantization front-end for Core AI, wrapping ``coreai_opt``.

    Standard PT2E lifecycle: :meth:`prepare` (exports + inserts fake-quant),
    :meth:`calibration_mode` / :meth:`training_mode`, then :meth:`convert`.

    Unlike stock PT2E (where ``convert`` yields a backend-agnostic Q/DQ graph),
    :meth:`convert` here returns the Core AI quantized graph (``coreai::`` ops),
    because ``coreai_opt``'s fake-quant has no meaningful generic Q/DQ stage and
    the full rewrite must complete before ``torch.export``.

    Args:
        model: the ``nn.Module`` to quantize.
        config: a ``coreai_opt.quantization.QuantizerConfig``. Defaults to
            ``coreai_opt``'s default (int8 weight + activation) when ``None``.
    """

    def __init__(self, model: nn.Module, config: Optional[Any] = None) -> None:
        from coreai_opt.quantization import Quantizer, QuantizerConfig

        self._quantizer = Quantizer(model, config or QuantizerConfig())
        self._prepared: Optional[fx.GraphModule] = None

    def prepare(
        self,
        example_inputs: Sequence[Any],
        dynamic_shapes: Any = None,
    ) -> fx.GraphModule:
        """PT2E prepare: export + insert fake-quant. Returns the prepared graph."""
        self._prepared = self._quantizer.prepare(
            tuple(example_inputs), dynamic_shapes=dynamic_shapes
        )
        return self._prepared

    def _require_prepared(self, method: str) -> None:
        if self._prepared is None:
            raise RuntimeError(f"Call prepare() before {method}().")

    def calibration_mode(self):
        """Context manager to collect activation statistics on calibration data."""
        self._require_prepared("calibration_mode")
        return self._quantizer.calibration_mode()

    def training_mode(self):
        """Context manager for quantization-aware training."""
        self._require_prepared("training_mode")
        return self._quantizer.training_mode()

    def convert(self) -> fx.GraphModule:
        """Produce the export-ready, fully quantized Core AI graph.

        Runs the entire ``coreai_opt`` ``finalize(CoreAI)`` (``convert_pt2e`` +
        conv/bn fold + Q/DQ -> ``coreai::`` rewrite + kv-cache relocation).  The
        result has no fake-quant, so it can be strict-``torch.export``ed and then
        lowered via ``to_edge_transform_and_lower([CoreAIPartitioner()])``.
        """
        from coreai_opt.common import ExportBackend

        self._require_prepared("convert")
        return self._quantizer.finalize(backend=ExportBackend.CoreAI)
