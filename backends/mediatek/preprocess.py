# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import contextlib
import struct

from typing import final, List

import mtk_converter
import mtk_neuron
import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

SKIP_COMPILE_SPEC_KEYS = {"ImportForever"}
REQUIRED_COMPILE_SPEC_KEYS = {"platform-config"}
SUPPORTED_PLATFORM_CONFIGS = {"mt6989", "mt6991"}


@final
class NeuropilotBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:

        # Validate CompileSpec settings
        compile_spec_keys = [spec.key for spec in module_compile_spec]
        if len(compile_spec_keys) != len(set(compile_spec_keys)):
            raise RuntimeError(
                "Unsupported duplicated keys in the CompileSpec settings."
            )
        if not REQUIRED_COMPILE_SPEC_KEYS.issubset(set(compile_spec_keys)):
            raise RuntimeError(
                "Following keys are required in the CompileSpec settings: {}."
                "".format(REQUIRED_COMPILE_SPEC_KEYS)
            )
        platform = [
            spec.value.decode("utf-8")
            for spec in module_compile_spec
            if spec.key == "platform-config"
        ][0]
        if platform not in SUPPORTED_PLATFORM_CONFIGS:
            raise ValueError(
                "Unsupported value of platform-config CompileSpec. Given {} but expected to be one "
                "of {}.".format(platform, SUPPORTED_PLATFORM_CONFIGS)
            )

        name_to_node_mappings = {node.name: node for node in edge_program.graph.nodes}
        input_names = edge_program.graph_signature.user_inputs
        output_names = edge_program.graph_signature.user_outputs
        fp_input_indices = [
            idx
            for idx, name in enumerate(input_names)
            if name_to_node_mappings[name].meta["val"].dtype == torch.float32
        ]
        fp_output_indices = [
            idx
            for idx, name in enumerate(output_names)
            if name_to_node_mappings[name].meta["val"].dtype == torch.float32
        ]

        compile_options = ["--relax-fp32", "--opt=3"]
        for spec in module_compile_spec:
            if spec.key in SKIP_COMPILE_SPEC_KEYS:
                continue
            if spec.value == b"":
                compile_options.append(f"--{spec.key}")
            else:
                value = spec.value.decode("utf-8")
                compile_options.append(f"--{spec.key}={value}")

        converter = mtk_converter.PyTorchV2Converter.from_exported_program(edge_program)
        converter.quantize = True
        converter.input_quantization_bitwidths = None
        converter.allow_missing_quantization_ranges = True
        converter.prepend_input_quantize_ops = True
        converter.prepend_input_quantize_ops_indices = fp_input_indices
        converter.append_output_dequantize_ops = True
        converter.append_output_dequantize_ops_indices = fp_output_indices
        with contextlib.redirect_stdout(None):
            mlir_str = converter.convert_to_mlir()
            model_bytes = mtk_neuron.compile(mlir_str, " ".join(compile_options))

        num_inputs = len(input_names)
        num_outputs = len(output_names)
        header = struct.pack("<BIII", 1, num_inputs, num_outputs, len(model_bytes))
        return PreprocessResult(processed_bytes=bytes(header + model_bytes))
