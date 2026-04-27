# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os

try:
    from eiq_neutron_sdk import neutron_converter, neutron_library_utils
except ImportError:
    raise RuntimeError(
        "eIQ Neutron SDK not found. To install it, run 'examples/nxp/setup.sh'."
    )


def convert_unsafe(neutron_converter, tflite_model, cctx, queue):
    """
    Run neutron_converter on given tflite_model with compilation context cctx.
    This routine is supposed to run in a separate process.
    If properly finished, the output queue contains the converted model,
    otherwise the neutron_converter exits and the output queue is empty.
    """
    model_converted = neutron_converter.convertModel(list(tflite_model), cctx)
    queue.put(model_converted)


class NeutronConverterManager:
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

    def get_converter(self):
        return neutron_converter

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

    def convert(
        self,
        tflite_model: bytes,
        target: str,
        delegation_tag: str,
        fetch_constants_to_sram: bool = False,
        use_new_flow_neutron_c: bool = False,
    ) -> bytes:
        """
        Call Neutron Converter.

        :param tflite_model: A generic TFLite model to be converted.
        :param target: The target platform.
        :param delegation_tag: The delegation tag of model partition.
        :param fetch_constants_to_sram: Add microcode that fetches weights from external memory.
        :param use_new_flow_neutron_c: Enable experimental MLIR-based flow for Neutron-C with improved INT8 operator support.
        This allows running models which do not fit into SRAM. Applies to Neutron-C only (microcontrollers).

        :return: TFLite model with Neutron microcode as bytes.
        """
        # Neutron converter crashes if we provide invalid target -> verify.
        self.verify_target(target)

        cctx = neutron_converter.CompilationContext()
        cctx.targetOpts = neutron_converter.getNeutronTarget(target)
        cctx.compilationOpts.minNumOpsPerGraph = 1
        cctx.compilationOpts.excludeGraphPasses = (
            "HoistSliceAboveTranspose,MergeTranspose"
        )
        cctx.compilationOpts.fetchConstantsToSRAM = fetch_constants_to_sram
        cctx.compilationOpts.dumpKernelSelectionCode = self.dump_kernel_selection_code
        if hasattr(cctx.compilationOpts, "useNewFlowNeutronC"):
            cctx.compilationOpts.useNewFlowNeutronC = use_new_flow_neutron_c

        # Try to use multiprocessing for isolation, but fall back to direct execution
        # if the environment doesn't support it (e.g., in sandcastle/build environments)
        try:
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(logging.WARNING)
            queue = multiprocessing.Manager().Queue()

            process = multiprocessing.Process(
                target=convert_unsafe,
                args=(neutron_converter, tflite_model, cctx, queue),
            )
            process.start()
            process.join()  # waits until the subprocess is complete

            if queue.empty():  # signals the unsafe task did not run till the end
                raise RuntimeError(
                    f"Neutron converter module terminated unexpectedly with exit code {process.exitcode}"
                )

            model_converted = queue.get()
            process.close()
        except (EOFError, OSError) as e:
            # Multiprocessing failed (likely due to environment restrictions)
            # Fall back to direct execution
            logging.warning(
                f"Multiprocessing not available ({e}), running neutron converter directly"
            )
            model_converted = neutron_converter.convertModel(list(tflite_model), cctx)
        if self.dump_kernel_selection_code:
            self._rename_partition_kernel_selection_file(delegation_tag)

        return bytes(model_converted)
