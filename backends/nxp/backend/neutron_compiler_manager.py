# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import warnings
from multiprocessing.connection import wait

try:
    from eiq_neutron_sdk import neutron_compiler, neutron_library_utils

    _USING_NEUTRON_COMPILER = True
except ImportError:
    try:
        from eiq_neutron_sdk import (
            neutron_converter as neutron_compiler,
            neutron_library_utils,
        )

        _USING_NEUTRON_COMPILER = False
        warnings.warn(
            "The support for eIQ Neutron SDK <= 3.2.2 will be removed in future releases.",
            DeprecationWarning,
            stacklevel=2,
        )
    except ImportError:
        raise RuntimeError(
            "eIQ Neutron SDK not found. To install it, run 'examples/nxp/setup.sh'."
        )


def _build_compilation_context(compilation_opts):
    """Build a CompilationContext from a plain dict of options."""
    cctx = neutron_compiler.CompilationContext()
    cctx.targetOpts = neutron_compiler.getNeutronTarget(compilation_opts["target"])
    cctx.compilationOpts.minNumOpsPerGraph = compilation_opts["minNumOpsPerGraph"]
    cctx.compilationOpts.excludeGraphPasses = compilation_opts["excludeGraphPasses"]
    cctx.compilationOpts.fetchConstantsToSRAM = compilation_opts["fetchConstantsToSRAM"]
    cctx.compilationOpts.dumpKernelSelectionCode = compilation_opts[
        "dumpKernelSelectionCode"
    ]
    if compilation_opts["useProfiling"]:
        if not _USING_NEUTRON_COMPILER:
            raise RuntimeError(
                "Profiling requires eIQ Neutron SDK >= 3.2.1. "
                "The installed SDK (neutron_converter-based, <= 3.2.0) does not provide neutronGetSdkVersion(); "
                "the ExecuTorch runtime will fail to link. Upgrade the SDK."
            )
        cctx.compilationOpts.useProfiling = True
        cctx.compilationOpts.dumpAfterImport = "console"
        cctx.compilationOpts.dumpAfterGenerate = "console"
        cctx.compilationOpts.verbose = True
        cctx.compilationOpts.dumpMicrocode = True

    return cctx


def compile_unsafe(tflite_model, compilation_opts, send_conn):
    """
    Run neutron_compiler on the given tflite_model in a separate process.
    On success the compiled model is sent through send_conn.
    On a compiler crash (SIGSEGV / C-level exit()) the process dies without
    sending anything, which the parent detects via the subprocess timeout.
    """
    cctx = _build_compilation_context(compilation_opts)
    if _USING_NEUTRON_COMPILER:
        model_compiled = neutron_compiler.compileModel(list(tflite_model), cctx)
    else:
        model_compiled = neutron_compiler.convertModel(list(tflite_model), cctx)
    send_conn.send(model_compiled)
    send_conn.close()


# Maximum seconds to wait for the compiler subprocess to send its result.
# Set conservatively large so that slow but valid compilations are never killed.
# The timeout only fires when the child is stuck (e.g. a native crash handler
# that does not terminate), which would otherwise hang the caller indefinitely
# when tests are run in parallel.
_COMPILER_SUBPROCESS_TIMEOUT_S = 300  # 5 minutes.


class NeutronCompilerManager:
    """
    Manager for conversion of TFLite model in flatbuffers format into TFLite model that
    contains NeutronGraph nodes.
    """

    def __init__(
        self,
        dump_kernel_selection_code: bool = False,
    ):
        self.dump_kernel_selection_code = dump_kernel_selection_code

    @staticmethod
    def _rename_partition_kernel_selection_file(delegation_tag):
        try:
            base_name = "_kernel_selection.c"
            os.rename(base_name, f"_kernel_selection_{delegation_tag}.c")
        except OSError:
            logging.error("Failed to rename partition kernel selection file.")

    def get_compiler(self):
        return neutron_compiler

    def get_library_utils(self):
        return neutron_library_utils

    def verify_target(self, target: str):
        if not neutron_library_utils.isNeutronTarget(target):
            valid_targets = [
                target.name for target in neutron_library_utils.getNeutronTargets()
            ]
            raise ValueError(
                f"Target `{target}` is not a valid target. Must be one of `{valid_targets}`."
            )

    def compile(
        self,
        tflite_model: bytes,
        target: str,
        delegation_tag: str,
        fetch_constants_to_sram: bool = False,
        use_profiling: bool = False,
    ) -> bytes:
        """
        Call Neutron Compiler.

        :param tflite_model: A generic TFLite model to be compiled.
        :param target: The target platform.
        :param delegation_tag: The delegation tag of model partition.
        :param fetch_constants_to_sram: Add microcode that fetches weights from external memory.
        :param use_profiling: Use profiling for neutron delegated model.
        This allows running models which do not fit into SRAM. Applies to Neutron-C only (microcontrollers).

        :return: TFLite model with Neutron microcode as bytes.
        """
        self.verify_target(target)

        compilation_opts = {
            "target": target,
            "minNumOpsPerGraph": 1,
            "excludeGraphPasses": "HoistSliceAboveTranspose,MergeTranspose",
            "fetchConstantsToSRAM": fetch_constants_to_sram,
            "dumpKernelSelectionCode": self.dump_kernel_selection_code,
            "useProfiling": use_profiling,
        }

        # Run the compiler in a subprocess to isolate crashes (the Neutron SDK can
        # call exit() or segfault). A raw Pipe is used so that closing send_conn in
        # the parent after fork makes the child the sole writer; a clean crash then
        # produces EOF on recv_conn. For crashes where the native signal handler
        # hangs, the subprocess timeout detects the stuck child and kills it.
        try:
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(logging.WARNING)

            recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
            process = multiprocessing.Process(
                target=compile_unsafe,
                args=(tflite_model, compilation_opts, send_conn),
            )
            process.start()
            send_conn.close()  # child is now the sole writer

            ready = wait([recv_conn], timeout=_COMPILER_SUBPROCESS_TIMEOUT_S)
            if not ready:
                if process.is_alive():
                    process.kill()
                process.join()
                recv_conn.close()
                raise RuntimeError(
                    f"Neutron compiler subprocess did not respond within "
                    f"{_COMPILER_SUBPROCESS_TIMEOUT_S} s and was killed "
                    f"(exit code {process.exitcode})"
                )

            try:
                model_compiled = recv_conn.recv()
            except EOFError:
                model_compiled = None
            finally:
                recv_conn.close()

            process.join()

            if model_compiled is None or process.exitcode != 0:
                raise RuntimeError(
                    f"Neutron compiler module terminated unexpectedly with exit code {process.exitcode}"
                )

            process.close()
        except (OSError, TypeError) as e:
            # Multiprocessing not available (e.g. restricted sandbox environment);
            # fall back to running the compiler directly in the current process.
            logging.warning(
                f"Multiprocessing not available ({e}), running neutron compiler directly"
            )
            cctx = _build_compilation_context(compilation_opts)
            if _USING_NEUTRON_COMPILER:
                model_compiled = neutron_compiler.compileModel(list(tflite_model), cctx)
            else:
                model_compiled = neutron_compiler.convertModel(list(tflite_model), cctx)

        if self.dump_kernel_selection_code:
            self._rename_partition_kernel_selection_file(delegation_tag)

        return bytes(model_compiled)
