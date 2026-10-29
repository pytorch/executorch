# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class QuantizerAdapter(Protocol):
    """Protocol for quantization operations.

    Wraps external quantization APIs (prepare_pt2e, convert_pt2e)
    behind an injectable interface for testability.

    .. note::
        ``export_model``, ``prepare_pt2e``, ``init_encodings``, and
        ``convert_pt2e`` each operate on one graph module. Their module arguments
        are scalar values, never component or graph maps.

        All routing lives in the **quantization strategy**: it fans out over the
        components and graph variants, decides which single graph is the one to
        quantize (versus the deployed graphs that are only run once for their
        observers), collects those graphs, drives their quantization, and then
        propagates the resulting scales / zero points onto the deployed graphs.
        ``calibrate`` is the exception:
        it receives the single-level ``{component: module}`` map so model-family
        inference can drive cross-component calibration. There is no graph axis
        because the strategy has already selected one calibration graph per
        component.
    """

    def export_model(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Export a single graph module using torch.export.

        Args:
            module: One graph module, taken from the value of the strategy's
                ``{component: module}`` map. It is not a mapping.
            example_inputs: Positional example inputs describing this graph's
                export signature.

        Returns:
            The exported module (e.g., ExportedProgram.module()).
        """
        ...

    def prepare_pt2e(
        self,
        module: Any,
        quantizer: Any,
    ) -> Any:
        """Prepare a single exported module for PT2E quantization.

        Args:
            module: One exported graph module, not a component or graph map.
            quantizer: The configured quantizer.

        Returns:
            The annotated module with observers inserted.
        """
        ...

    def init_encodings(
        self,
        module: Any,
        example_inputs: Any,
    ) -> Any:
        """Initialize a graph's observers with a single dummy forward.

        graphs (AR-1 decode / AR-N prefill) are not truly calibrated;
        they run once on their own example inputs so their observers'
        placeholders are populated before the encoding-override step copies the
        real encodings in from the calibration graph.

        Args:
            module: One annotated deployed graph module, not a map.
            example_inputs: This graph's positional example-input tuple.

        Returns:
            The module after the dummy forward.
        """
        ...

    def calibrate(
        self,
        modules: Any,
        calibration_data: Iterable[Any],
        **kwargs: Any,
    ) -> Any:
        """Run true calibration over the quantization graph.

        Drives ``calibration_data`` through ``module``. Model-specific adapters
        may use extra kwargs such as ``inference`` (a ``ModelInference`` bound
        to the calibration graph); generic adapters can directly call the
        module.

        Args:
            modules: ``{component: module}`` for the selected calibration graphs.
                The graph axis has already been removed because each component
                contributes only its calibration graph.
            calibration_data: ``{component: DataLoader}`` of corpus-backed
                calibration batches.
            **kwargs: Extra adapter-specific options, such as optional
                ``inference`` for model-specific adapters.

        Returns:
            The calibrated module.
        """
        ...

    def convert_pt2e(
        self,
        module: Any,
    ) -> Any:
        """Convert a single calibrated module to a quantized module.

        Args:
            module: One calibrated graph module, not a component or graph map.

        Returns:
            The quantized module with fake quantize nodes replaced.
        """
        ...
