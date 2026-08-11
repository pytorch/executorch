# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core AI palettization front-end for ExecuTorch.

Palettization replaces a weight with a small learned lookup table plus an index
per element, rather than a scale and zero-point. It sits alongside
quantization::

    prepare -> (optional sensitivity calibration) -> finalize -> export

Unlike :class:`~executorch.backends.apple.coreai.quantizer.quantizer.CoreAIQuantizer`,
:meth:`finalize` returns an eager ``nn.Module`` whose weights carry a
parametrization, not an fx graph. The compression becomes a ``lut_to_dense`` op
only once the model is exported, so nothing here needs to run before
``torch.export`` for the backend's sake; the ordinary
``to_edge_transform_and_lower`` path lowers it.

Palettization is weight-only by construction, so there is no activation
counterpart to configure.

``coreai_opt`` (the ``coreai-optimization`` package) is imported lazily so this
module stays importable in environments without it installed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from torch import nn


class CoreAIPalettizer:
    """K-means palettization front-end for Core AI, wrapping ``coreai_opt``.

    Args:
        model: the ``nn.Module`` to palettize.
        config: a ``coreai_opt.palettization.KMeansPalettizerConfig``. Defaults
            to ``coreai_opt``'s own default when ``None``. Ready-made
            configurations are available as
            ``KMeansPalettizerConfig.presets.w4()``, ``.w6()`` and ``.w8()``.
    """

    def __init__(self, model: nn.Module, config: Optional[Any] = None) -> None:
        from coreai_opt.palettization import KMeansPalettizer

        self._palettizer = KMeansPalettizer(model, config)
        self._prepared: Optional[nn.Module] = None

    def prepare(
        self,
        example_inputs: Sequence[Any],
        sensitivity_path: Optional[str] = None,
        num_workers: int = 1,
    ) -> nn.Module:
        """Cluster the weights and attach the palette parametrizations.

        Args:
            example_inputs: Sample inputs used to trace the model.
            sensitivity_path: Sensitivities saved by a previous
                :meth:`calibration_mode` run, to weight the clustering.
            num_workers: Parallelism for clustering, which dominates the cost
                on large models.
        """
        self._prepared = self._palettizer.prepare(
            tuple(example_inputs),
            sensitivity_path=sensitivity_path,
            num_workers=num_workers,
        )
        return self._prepared

    def calibration_mode(self, *, loss_fn: Callable, sensitivity_path=None):
        """Context manager for sensitivity-weighted clustering (SqueezeLLM).

        Optional, and unlike the quantizer's calibration this is not needed for
        a plain run: k-means clusters the weights, which requires no data.
        Supplying a loss function collects squared gradients as per-element
        sensitivities so clustering favors the weights that matter most.
        """
        return self._palettizer.calibration_mode(
            loss_fn=loss_fn, sensitivity_path=sensitivity_path
        )

    def training_mode(self):
        """Context manager for palettization-aware training."""
        return self._palettizer.training_mode()

    def finalize(self, mmap_dir=None) -> nn.Module:
        """Produce the export-ready palettized model.

        Returns an eager ``nn.Module``, not a graph: the palette surfaces as a
        ``lut_to_dense`` op when the result is exported.

        Args:
            mmap_dir: Directory to memory-map large palettes through, rather
                than holding them in memory.
        """
        from coreai_opt.common import ExportBackend

        if self._prepared is None:
            raise RuntimeError("Call prepare() before finalize().")
        return self._palettizer.finalize(
            backend=ExportBackend.CoreAI, mmap_dir=mmap_dir
        )
