# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#
"""Ahead-of-time Arm Ethos-U backend built on the shared TOSA pipeline."""

import contextlib
import logging
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import final, List

from executorch.backends.arm.arm_vela import vela_compile, VelaCompileResult
from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec

from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

# debug functionality
logger = logging.getLogger(__name__)

EthosUPreprocessObserver = Callable[
    [bytes, tuple[str, ...], VelaCompileResult],
    object | None,
]

_PREPROCESS_OBSERVER: ContextVar[EthosUPreprocessObserver | None] = ContextVar(
    "ethosu_preprocess_observer", default=None
)


@contextlib.contextmanager
def observe_ethosu_preprocess(
    observer: EthosUPreprocessObserver,
) -> Iterator[None]:
    """Observe exact TOSA-to-Vela compilation in the current execution context.

    Args:
        observer: Callback receiving the exact TOSA bytes, immutable compiler
            arguments, and Vela result. It returns optional delegate metadata.

    """
    if not callable(observer):
        raise TypeError("Ethos-U preprocess observer must be callable")
    token = _PREPROCESS_OBSERVER.set(observer)
    try:
        yield
    finally:
        _PREPROCESS_OBSERVER.reset(token)


@final
class EthosUBackend(BackendDetails):
    """BackendDetails subclass for delegation to Ethos-U.

    Deduce the TOSA lowering from the compile spec list by filtering out the
    compile spec values that are of interest for the TOSABackend.

    """

    @staticmethod
    def _compile_tosa_flatbuffer(
        tosa_flatbuffer: bytes,
        compile_spec: EthosUCompileSpec,
    ) -> VelaCompileResult:
        """Compile a TOSA flatbuffer into a target-specific binary stream.

        Args:
            tosa_flatbuffer (bytes): Serialized TOSA graph produced by
                ``TOSABackend``.
            compile_spec (EthosUCompileSpec): Compile specification providing
                Vela flags and intermediate paths.
        Returns:
            VelaCompileResult: Binary stream and external payloads from Vela.

        """
        compile_flags = compile_spec.compiler_flags

        if len(compile_flags) == 0:
            # Not testing for compile_flags correctness here, just that they are
            # present. The compiler will give errors if they are not valid.
            raise RuntimeError(
                "compile_flags are required in the CompileSpec list for EthosUBackend"
            )

        # Vela tooling only supports flatbuffers up to 2 GiB.
        max_flatbuffer_size = 2 * 1024 * 1024 * 1024
        flatbuffer_size = len(tosa_flatbuffer)
        if flatbuffer_size > max_flatbuffer_size:
            raise RuntimeError(
                "TOSA flatbuffer is too large for Vela "
                f"({flatbuffer_size} bytes > {max_flatbuffer_size} bytes limit)."
            )

        # Pass on the TOSA flatbuffer to the vela compiler.
        return vela_compile(
            tosa_flatbuffer,
            compile_flags,
            verbose=logger.getEffectiveLevel() <= logging.INFO,
            intermediate_path=compile_spec._get_intermediate_path(),
            block_placements=(
                compile_spec.external_block_placements.to_block_placements()
            ),
            max_scratch_size=compile_spec.max_scratch_size,
        )

    @staticmethod
    def _compile_with_observer(
        tosa_flatbuffer: bytes,
        compile_spec: EthosUCompileSpec,
    ) -> tuple[VelaCompileResult, object | None]:
        compile_result = EthosUBackend._compile_tosa_flatbuffer(
            tosa_flatbuffer, compile_spec
        )
        observer = _PREPROCESS_OBSERVER.get()
        if observer is None:
            return compile_result, None

        delegate_metadata = observer(
            tosa_flatbuffer,
            tuple(compile_spec.compiler_flags),
            compile_result,
        )
        return compile_result, delegate_metadata

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Lower the exported program and compile it for an Ethos-U target.

        Args:
            edge_program (ExportedProgram): Program to lower to Ethos-U.
            compile_specs (List[CompileSpec]): Serialized Ethos-U compile specs
                supplied by the frontend.

        Returns:
            PreprocessResult: Result containing the compiled Ethos-U binary.

        """
        logger.info(f"{EthosUBackend.__name__} preprocess")

        compile_spec = EthosUCompileSpec._from_list(compile_specs)
        # deduce TOSA compile_spec from Ethos-U compile spec. We get a new
        # compile spec list, containing only elements relevant for the
        # TOSABackend.
        tosa_compile_spec = TOSABackend.filter_tosa_compile_specs(compile_spec)

        # Backends doesn't allow inheritance, as stated in comments in exir/backend/backend_api.py
        # ('All backend implementation are final...'), so use composition instead.
        # preprocess returns the serialized TOSA flatbuffer in .processed_bytes,
        # which can be passed on to next compilation step.
        tosa_preprocess = TOSABackend._preprocess(edge_program, tosa_compile_spec)

        compile_result, delegate_metadata = EthosUBackend._compile_with_observer(
            tosa_preprocess.processed_bytes, compile_spec
        )
        if not compile_result.external_blocks:
            return PreprocessResult(
                processed_bytes=compile_result.processed_bytes,
                _delegate_info_meta=delegate_metadata,
            )

        data_store = NamedDataStore()
        for external_block in compile_result.external_blocks:
            data_store.add_named_data(
                external_block.key,
                external_block.payload,
                alignment=external_block.alignment,
                external_tag=external_block.placement,
            )
        return PreprocessResult(
            processed_bytes=compile_result.processed_bytes,
            data_store_output=data_store.get_named_data_store_output(),
            _delegate_info_meta=delegate_metadata,
        )
