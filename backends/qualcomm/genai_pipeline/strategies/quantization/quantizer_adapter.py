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

    Wraps external quantization APIs (make_quantizer, prepare_pt2e, convert_pt2e)
    behind an injectable interface for testability.

    .. note::
        These methods operate on a **single graph**, mirroring the underlying
        ``torchao`` PT2E APIs, and adapters stay a thin 1:1 wrapper over them.

        For models exported as several graphs from the same weights -- e.g. a
        hybrid decoder -- only one graph is actually quantized: a dedicated
        full-auto-regressive calibration graph, which yields the best activation
        statistics but is never deployed. Its scales and zero points are then
        propagated onto the deployed graphs (AR-N prefill, AR-1 decode), which do
        not run PT2E themselves. That propagation is inherently cross-graph, so
        it cannot live in a per-graph method here; the **quantization strategy**
        sequences it via a separate reconciliation hook after this adapter
        returns.
    """

    def make_quantizer(
        self,
        quant_dtype: Any = None,
        backend: Any = None,
        soc_model: Any = None,
        quant_recipe: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create a QNN quantizer with the given configuration.

        Every argument defaults to ``None`` so that callers can omit any of them
        and let the implementation -- or the API it wraps -- supply the default.
        In particular an omitted ``quant_dtype`` must not be forwarded, so the
        underlying ``make_quantizer`` default applies rather than being shadowed.

        Args:
            quant_dtype: Quantization data type (e.g., QuantDtype.use_8a8w).
                ``None`` selects the implementation's default.
            backend: QNN backend type (HTP, GPU, LPAI).
            soc_model: Target SoC chipset.
            quant_recipe: Optional quantization recipe. Applied to the
                constructed quantizer (``QnnQuantizer.set_recipe``) rather than
                passed to ``make_quantizer``, which takes no such argument.
            **kwargs: Additional quantizer options (per_channel, observers, etc.).

        Returns:
            A configured quantizer instance.
        """
        ...

    def export_model(
        self,
        model: Any,
        sample_input: Any,
    ) -> Any:
        """Export the model using torch.export.

        Args:
            model: The nn.Module to export.
            sample_input: Sample input tuple for tracing.

        Returns:
            The exported model (e.g., ExportedProgram.module()).
        """
        ...

    def prepare_pt2e(
        self,
        model: Any,
        quantizer: Any,
    ) -> Any:
        """Prepare the model for PT2E quantization (insert observers).

        Args:
            model: The exported model.
            quantizer: The configured quantizer.

        Returns:
            The annotated model with observers inserted.
        """
        ...

    def calibrate(
        self,
        model: Any,
        calibration_data: Iterable[Any],
    ) -> Any:
        """Run calibration data through the annotated model.

        Implementations needing a non-trivial procedure -- e.g. autoregressive
        LLM calibration, where each step's input depends on the previous step's
        output -- should override this method rather than encoding the procedure
        in ``calibration_data``.

        Args:
            model: The annotated model with observers.
            calibration_data: Any ``Iterable[Tuple[Tensor, ...]]``, including a
                plain list or a ``DataLoader``.

        Returns:
            The calibrated model.
        """
        ...

    def convert_pt2e(
        self,
        model: Any,
    ) -> Any:
        """Convert the calibrated model to a quantized model.

        Args:
            model: The calibrated model.

        Returns:
            The quantized model with fake quantize nodes replaced.
        """
        ...
