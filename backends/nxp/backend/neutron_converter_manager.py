# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import pkgutil

from executorch.backends.nxp.backend.ir.converter.node_converter import Target


class NeutronConverterManager:
    """
    Manager for conversion of TFLite model in flatbuffers format into TFLite model that
    contains NeutronGraph nodes.
    """

    _supported_target_names = [Target.RT700.value]

    def convert(
        self, tflite_model: bytes, target: str, neutron_converter_flavor: str
    ) -> bytes:
        # Neutron converter crashes if we provide invalid target -> verify.
        if target not in self._supported_target_names:
            raise RuntimeError(
                f"Target '{target}' is not supported by NeutronConverterManager."
            )

        neutron_converter_modules = [
            module.name
            for module in pkgutil.iter_modules()
            if module.name.startswith("neutron_converter")
        ]

        requested_module_name = f"neutron_converter_{neutron_converter_flavor}"
        if requested_module_name not in neutron_converter_modules:
            if len(neutron_converter_modules) > 0:
                raise RuntimeError(
                    f"Neutron Converter module with flavor '{neutron_converter_flavor}' "
                    f"not found. Available modules: {neutron_converter_modules}."
                )
            else:
                raise RuntimeError(
                    f"Neutron Converter module with flavor '{neutron_converter_flavor}' "
                    f"not found. Install 'neutron_converter_[flavor]' Python package."
                )

        neutron_converter = importlib.import_module(
            f"{requested_module_name}.neutron_converter"
        )

        cctx = neutron_converter.CompilationContext()
        cctx.targetOpts = neutron_converter.getNeutronTarget(target)
        # New switch since Neutron Converter SDK_25.06
        cctx.compilationOpts.minNumOpsPerGraph = 1
        model_converted = neutron_converter.convertModel(list(tflite_model), cctx)

        return bytes(model_converted)
