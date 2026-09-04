# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os

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
    if (
        hasattr(cctx.compilationOpts, "useProfiling")
        and compilation_opts["useProfiling"]
    ):
        cctx.compilationOpts.useProfiling = compilation_opts["useProfiling"]
        cctx.compilationOpts.dumpAfterImport = "console"
        cctx.compilationOpts.dumpAfterGenerate = "console"
        cctx.compilationOpts.verbose = compilation_opts["useProfiling"]

    return cctx


def compile_unsafe(tflite_model, compilation_opts, queue):
    """
    Run neutron_compiler on given tflite_model with the provided compilation options.
    This routine is supposed to run in a separate process.
    If properly finished, the output queue contains the compiled model,
    otherwise the neutron_compiler exits and the output queue is empty.
    """
    cctx = _build_compilation_context(compilation_opts)
    if _USING_NEUTRON_COMPILER:
        model_compiled = neutron_compiler.compileModel(list(tflite_model), cctx)
    else:
        model_compiled = neutron_compiler.convertModel(list(tflite_model), cctx)
    queue.put(model_compiled)


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
        # Neutron compiler crashes if we provide invalid target -> verify.
        self.verify_target(target)

        compilation_opts = {
            "target": target,
            "minNumOpsPerGraph": 1,
            "excludeGraphPasses": "HoistSliceAboveTranspose,MergeTranspose",
            "fetchConstantsToSRAM": fetch_constants_to_sram,
            "dumpKernelSelectionCode": self.dump_kernel_selection_code,
            "useProfiling": use_profiling,
        }

        # Try to use multiprocessing for isolation, but fall back to direct execution
        # if the environment doesn't support it (e.g., in sandcastle/build environments)
        try:
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(logging.WARNING)
            queue = multiprocessing.Manager().Queue()

            process = multiprocessing.Process(
                target=compile_unsafe,
                args=(tflite_model, compilation_opts, queue),
            )
            process.start()
            process.join()  # waits until the subprocess is complete

            if queue.empty():  # signals the unsafe task did not run till the end
                raise RuntimeError(
                    f"Neutron compiler module terminated unexpectedly with exit code {process.exitcode}"
                )

            model_compiled = queue.get()
            process.close()
        except (EOFError, OSError, TypeError) as e:
            # Multiprocessing failed (likely due to environment restrictions)
            # Fall back to direct execution
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
