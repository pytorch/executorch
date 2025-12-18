# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import multiprocessing
import pkgutil


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
        neutron_converter_flavor: str = "SDK_25_09",
    ):

        neutron_converter_modules = [
            module.name
            for module in pkgutil.iter_modules()
            if module.name.startswith("neutron_converter")
            or module.name == "eiq_neutron_sdk"
        ]

        if neutron_converter_flavor:
            requested_module_name = f"neutron_converter_{neutron_converter_flavor}"
            print(
                "Warning: The use of converter flavors will be deprecated. Use empty string to select 'eiq_neutron_sdk' module."
            )
        else:
            requested_module_name = "eiq_neutron_sdk"

        if requested_module_name not in neutron_converter_modules:
            if len(neutron_converter_modules) > 0:
                raise RuntimeError(
                    f"Neutron Converter module '{requested_module_name}' "
                    f"not found. Available modules: {neutron_converter_modules}."
                )
            else:
                raise RuntimeError(
                    f"Neutron Converter module '{requested_module_name}' "
                    f"not found. Install 'eiq_neutron_sdk' or 'neutron_converter_[flavor]' Python package."
                )

        self.neutron_converter = importlib.import_module(
            f"{requested_module_name}.neutron_converter"
        )
        self.neutron_library_utils = importlib.import_module(
            f"{requested_module_name}.neutron_library_utils"
        )

    def get_converter(self):
        return self.neutron_converter

    def get_library_utils(self):
        return self.neutron_library_utils

    def verify_target(self, target: str):
        if not self.neutron_library_utils.isNeutronTarget(target):
            valid_targets = [
                target.name for target in self.neutron_library_utils.getNeutronTargets()
            ]
            raise ValueError(
                f"Target `{target}` is not a valid target. Must be one of `{valid_targets}`."
            )

    def convert(self, tflite_model: bytes, target: str) -> bytes:
        # Neutron converter crashes if we provide invalid target -> verify.
        self.verify_target(target)

        cctx = self.neutron_converter.CompilationContext()
        cctx.targetOpts = self.neutron_converter.getNeutronTarget(target)
        cctx.compilationOpts.minNumOpsPerGraph = 1
        cctx.compilationOpts.excludeGraphPasses = (
            "HoistSliceAboveTranspose,MergeTranspose"
        )

        # Try to use multiprocessing for isolation, but fall back to direct execution
        # if the environment doesn't support it (e.g., in sandcastle/build environments)
        try:
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(logging.WARNING)
            queue = multiprocessing.Manager().Queue()

            process = multiprocessing.Process(
                target=convert_unsafe,
                args=(self.neutron_converter, tflite_model, cctx, queue),
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
            model_converted = self.neutron_converter.convertModel(
                list(tflite_model), cctx
            )

        return bytes(model_converted)
