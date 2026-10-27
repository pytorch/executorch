# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Construction of QNN compiler specifications.

``generate_qnn_executorch_compiler_spec`` needs backend options from a
per-backend helper (``generate_htp_compiler_spec`` /
``generate_gpu_compiler_spec``) and a ``QcomChipset`` enum rather than a SoC
name. ``QnnCompileSpecBuilder`` pairs those two calls behind one method so
callers do not repeat the branch, and holds the settings that are properties of
the target rather than of an individual graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec

logger = logging.getLogger(__name__)

BACKEND_HTP = "htp"
BACKEND_GPU = "gpu"

# Backends this builder can emit specs for. The QNN backend enum also has LPAI
# and DSP entries, but neither has a spec-generation path here yet.
SUPPORTED_BACKENDS = (BACKEND_HTP, BACKEND_GPU)


def resolve_soc_model(soc_model: Union[str, "QcomChipset"]) -> "QcomChipset":
    """Convert a SoC name to its ``QcomChipset`` enum member.

    ``PipelineContext`` carries ``soc_model`` as a string so that nothing in the
    pipeline has to import the QNN serialization schema; lowering needs the
    enum, because ``generate_qnn_executorch_compiler_spec`` validates against
    ``QcomChipset`` values and indexes a table with them. This is where that
    conversion happens.

    Args:
        soc_model: A SoC name as accepted on the command line (e.g. "SM8750"),
            or an already-resolved ``QcomChipset``.

    Returns:
        The matching ``QcomChipset`` member.

    Raises:
        ValueError: If the name matches no supported SoC. The message lists the
            valid names, since this is typically a user-supplied value.
    """
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import get_soc_to_chipset_map

    if isinstance(soc_model, QcomChipset):
        return soc_model

    chipset_map = get_soc_to_chipset_map()
    if soc_model not in chipset_map:
        raise ValueError(
            f"Unknown SoC model '{soc_model}'. Supported: " f"{sorted(chipset_map)}"
        )

    return chipset_map[soc_model]


def _backend_type_map() -> Dict[str, "QnnExecuTorchBackendType"]:
    """Map each backend name to its ``QnnExecuTorchBackendType`` member.

    ``QnnExecuTorchBackendType.__str__`` yields the backend name the command
    line uses ("htp", "gpu", ...), so the enum itself is the source of the valid
    names rather than a table here. ``kUndefinedBackend`` is excluded: it is the
    unset value, not a target.

    Returns:
        Backend name to enum member, for every selectable backend.
    """
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    return {
        str(member): member
        for member in QnnExecuTorchBackendType
        if member is not QnnExecuTorchBackendType.kUndefinedBackend
    }


def resolve_backend_type(
    backend: Union[str, "QnnExecuTorchBackendType"],
) -> "QnnExecuTorchBackendType":
    """Convert a backend name to its ``QnnExecuTorchBackendType`` member.

    Mirrors ``export_utils.get_backend_type``, but validates the name instead of
    raising ``AttributeError`` from a failed ``getattr``.

    Args:
        backend: A backend name, e.g. "htp", "gpu" or "lpai", or an
            already-resolved ``QnnExecuTorchBackendType``.

    Returns:
        The matching ``QnnExecuTorchBackendType`` member.

    Raises:
        ValueError: If the name matches no backend.
    """
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    if isinstance(backend, QnnExecuTorchBackendType):
        return backend

    backend_map = _backend_type_map()
    if backend not in backend_map:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: {sorted(backend_map)}"
        )

    return backend_map[backend]


class QnnCompileSpecBuilder:
    """Builds QNN compiler specifications for one compilation target.

    Holds the settings that describe the *target* -- which SoC, which backend,
    and whether the target is the x86 emulator -- so that per-graph calls to
    :meth:`build` only pass what varies between graphs.

    One instance describes one target. A model whose components go to different
    backends -- an encoder on GPU with the decoder on HTP -- needs one builder
    per backend rather than one builder reconfigured, since ``backend`` is
    constructor state.

    ``enable_x86_64`` is a constructor argument because the emulator's
    restrictions are properties of the target: it supports neither weight
    sharing nor shared buffers, so both default to off when it is set. Passing
    those explicitly to :meth:`build` still overrides the default.

    Example::

        builder = QnnCompileSpecBuilder(soc_model="SM8750")
        decoder_spec = builder.build(use_fp16=False, use_mha2sha=True)
        encoder_spec = builder.build(use_fp16=True)

    Args:
        soc_model: Target SoC, as a name or a ``QcomChipset``.
        backend: Target backend; one of :data:`SUPPORTED_BACKENDS`.
        enable_x86_64: Whether the target is the x86 emulator.

    Raises:
        ValueError: If ``soc_model`` or ``backend`` is not supported. Both are
            validated here rather than at ``build()`` time so that a bad target
            fails before any graph work.
    """

    def __init__(
        self,
        soc_model: Union[str, "QcomChipset"],
        backend: str = BACKEND_HTP,
        enable_x86_64: bool = False,
    ) -> None:
        if backend not in SUPPORTED_BACKENDS:
            # Distinguished from an unknown backend: the QNN enum has LPAI and
            # DSP members that ``resolve_backend_type`` resolves happily, they
            # just have no spec-generation path through this builder yet.
            raise ValueError(
                f"No compile spec generation path for backend '{backend}' yet. "
                f"Supported here: {list(SUPPORTED_BACKENDS)}"
            )

        self._soc_model = resolve_soc_model(soc_model)
        self._backend = backend
        self._enable_x86_64 = enable_x86_64

    @property
    def soc_model(self) -> "QcomChipset":
        """The resolved target SoC."""
        return self._soc_model

    @property
    def backend(self) -> str:
        """The target backend name."""
        return self._backend

    @property
    def backend_type(self) -> "QnnExecuTorchBackendType":
        """The target backend as a ``QnnExecuTorchBackendType``."""
        return resolve_backend_type(self._backend)

    def build(
        self,
        use_fp16: bool = False,
        use_multi_contexts: bool = False,
        use_weight_sharing: Optional[bool] = None,
        shared_buffer: Optional[bool] = None,
        online_prepare: bool = False,
        use_mha2sha: bool = False,
    ) -> List["CompileSpec"]:
        """Build the compiler specs for a single graph.

        Args:
            use_fp16: Compile for FP16 rather than quantized execution. HTP
                only; ignored for GPU, whose precision comes from the tensor
                data types.
            use_multi_contexts: Emit multiple contexts within one ``.pte`` so a
                single spill-fill allocation can be reused across them. Set
                this when the graph is sharded. Sizing that allocation is a
                post-lowering step and is not done here.
            use_weight_sharing: Share identical weights across the graphs of a
                multi-method ``.pte``. Defaults to on unless targeting the
                emulator, which does not support it.
            shared_buffer: Use a shared buffer for graph I/O. Defaults to on
                unless targeting the emulator, which does not support it.

                Note that this default is *not* ``ControlArgs.shared_buffer``,
                which defaults to ``False`` to match ``llama.py``'s parser. A
                caller driving this from a ``ControlArgs`` must therefore pass
                ``shared_buffer=control_args.shared_buffer`` explicitly;
                omitting it silently turns the feature on for a device target.
            online_prepare: Compose the QNN graph on device. Cannot be combined
                with ``use_multi_contexts``, which the underlying spec
                generation rejects.
            use_mha2sha: Convert multi-head attention to single-head attention.

        Returns:
            The compiler specs for one graph.
        """
        from executorch.backends.qualcomm.utils.utils import (
            generate_gpu_compiler_spec,
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
        )

        # The emulator supports neither feature; on device both are wanted.
        if use_weight_sharing is None:
            use_weight_sharing = not self._enable_x86_64
        if shared_buffer is None:
            shared_buffer = not self._enable_x86_64

        backend_options = self._build_backend_options(
            generate_gpu_compiler_spec=generate_gpu_compiler_spec,
            generate_htp_compiler_spec=generate_htp_compiler_spec,
            use_fp16=use_fp16,
            use_multi_contexts=use_multi_contexts,
            use_weight_sharing=use_weight_sharing,
        )

        logger.debug(
            "Building %s compile spec: soc=%s, fp16=%s, multi_contexts=%s, "
            "weight_sharing=%s, shared_buffer=%s, online_prepare=%s, mha2sha=%s",
            self._backend,
            self._soc_model.name,
            use_fp16,
            use_multi_contexts,
            use_weight_sharing,
            shared_buffer,
            online_prepare,
            use_mha2sha,
        )

        return generate_qnn_executorch_compiler_spec(
            soc_model=self._soc_model,
            backend_options=backend_options,
            shared_buffer=shared_buffer,
            online_prepare=online_prepare,
            use_mha2sha=use_mha2sha,
        )

    def _build_backend_options(
        self,
        generate_gpu_compiler_spec: Any,
        generate_htp_compiler_spec: Any,
        use_fp16: bool,
        use_multi_contexts: bool,
        use_weight_sharing: bool,
    ) -> Any:
        """Build the backend-specific options for the target backend.

        Args:
            generate_gpu_compiler_spec: The GPU options helper.
            generate_htp_compiler_spec: The HTP options helper.
            use_fp16: Whether to compile for FP16. HTP only.
            use_multi_contexts: Whether to emit multiple contexts. HTP only.
            use_weight_sharing: Whether to share weights across graphs.

        Returns:
            The backend options for the target backend.
        """
        if self._backend == BACKEND_HTP:
            return generate_htp_compiler_spec(
                use_fp16=use_fp16,
                use_multi_contexts=use_multi_contexts,
                use_weight_sharing=use_weight_sharing,
            )

        # GPU takes its precision from the tensor data types, so use_fp16 has no
        # equivalent; flag it rather than dropping a caller's request silently.
        if use_fp16:
            logger.warning(
                "use_fp16 has no effect for the GPU backend; precision follows "
                "the tensor data types"
            )
        return generate_gpu_compiler_spec(use_weight_sharing=use_weight_sharing)
