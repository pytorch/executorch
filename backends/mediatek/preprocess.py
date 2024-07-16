# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import contextlib
import json
import os
import shutil
import subprocess
import struct
import tempfile
import uuid

from pathlib import Path
from typing import final, List
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

import mtk_converter
import mtk_neuron

SKIP_COMPILE_SPEC_KEYS = {'ImportForever'}


@final
class NeuropilotBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls,
        edge_program: ExportedProgram,
        module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:

        header_version = 1
        num_inputs = len(edge_program.graph_signature.user_inputs)
        num_outputs = len(edge_program.graph_signature.user_outputs)

        # This default compile options are only for mt6989 SOC
        compile_options = ['--arch=mdla5.1,edpa1.0', '--relax-fp32', '--opt=3']
        for spec in module_compile_spec:
            if spec.key in SKIP_COMPILE_SPEC_KEYS:
                continue
            if spec.value is None:
                compile_options.append(f'--{spec.key}')
            else:
                value = spec.value.decode('utf-8')
                compile_options.append(f'--{spec.key}={value}')

        converter = mtk_converter.PyTorchV2Converter.from_exported_program(edge_program)
        converter.quantize = True
        converter.input_quantization_bitwidths = None
        converter.allow_missing_quantization_ranges = True
        converter.prepend_input_quantize_ops = True
        converter.append_output_dequantize_ops = True

        with contextlib.redirect_stdout(None):
            mlir_str = converter.convert_to_mlir()
            model_bytes = mtk_neuron.compile(mlir_str, ' '.join(compile_options))

        header = struct.pack('<BIII', 1, num_inputs, num_outputs, len(model_bytes))
        return PreprocessResult(processed_bytes=bytes(header + model_bytes))
