# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import contextlib
import struct

from typing import final, List, cast

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile

SKIP_COMPILE_SPEC_KEYS = {"ImportForever"}


@final
class OpenvinoBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:

        name_to_node_mappings = {node.name: node for node in edge_program.graph.nodes}
        input_names = edge_program.graph_signature.user_inputs
        output_names = edge_program.graph_signature.user_outputs
        args = []
        for node in edge_program.graph.nodes:
            if (node.target in input_names):
                args.append( node.meta["val"])

        input_shapes = []
        output_shapes = []

        compile_options = {}
        for spec in module_compile_spec:
            compile_options[spec.key] = spec.value.decode()

        compiled = openvino_compile(edge_program.module(), *args, options=compile_options)
        model_bytes = compiled.export_model()

        return PreprocessResult(processed_bytes=model_bytes)
