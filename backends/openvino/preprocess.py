# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import final, List

import torch
from executorch.backends.openvino._passes import DecomposeFloorDividePass

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass
from openvino import compile_model, opset13  # type: ignore[import-untyped]
from openvino.frontend import FrontEndManager  # type: ignore[import-untyped]
from openvino.frontend.pytorch.fx_decoder import (  # type: ignore[import-untyped]
    TorchFXPythonDecoder,
)
from openvino.preprocess import PrePostProcessor  # type: ignore[import-untyped]


@final
class OpenvinoBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:
        """
        Preprocesses the exported program and compiles it for the OpenVINO backend.

        Args:
            edge_program (ExportedProgram): The exported program representing the model.
            module_compile_spec (List[CompileSpec]): A list of compile specifications for the OpenVINO backend.

        Returns:
            PreprocessResult: The result of preprocessing, including the compiled model bytes.
        """
        for pass_cls in [DimOrderOpsRevertPass, DecomposeFloorDividePass]:
            result = pass_cls()(edge_program.graph_module)
            if result and result.graph_module:
                edge_program._graph_module = result.graph_module

        input_names = edge_program.graph_signature.user_inputs
        input_types = []
        input_shapes = []
        int_inputs = []
        for node in edge_program.graph.nodes:
            if node.target in input_names:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    input_types.append(val.dtype)
                    input_shapes.append(
                        [d if isinstance(d, int) else -1 for d in val.shape]
                    )
                elif isinstance(val, (int, torch.SymInt)) and not isinstance(val, bool):
                    int_inputs.append(len(input_types))
                    input_types.append(torch.int64)
                    input_shapes.append([])
                else:
                    raise RuntimeError(
                        f"OpenVINO backend: partition input {node.target} is a "
                        f"{type(val).__name__}; only tensors and ints are supported"
                    )

        compile_options = {}
        for spec in module_compile_spec:
            compile_options[spec.key] = spec.value.decode()

        # The converter chooses some lowerings (e.g. boolean-mask indexing) from
        # the input types, so they must be known before conversion starts.
        decoder = TorchFXPythonDecoder(
            edge_program.module(), input_shapes=input_shapes, input_types=input_types
        )
        frontend = FrontEndManager().load_by_framework("pytorch")
        ppp = PrePostProcessor(frontend.convert(frontend.load(decoder)))
        # The converter takes an int input, like a SymInt size, as a scalar, but
        # the runtime passes it as an int64 tensor of shape [1].
        for i in int_inputs:
            ppp.input(i).tensor().set_shape([1])
            ppp.input(i).preprocess().custom(lambda x: opset13.squeeze(x, 0).output(0))
        compiled = compile_model(ppp.build(), compile_options.get("device", "CPU"))
        model_bytes = compiled.export_model()

        return PreprocessResult(processed_bytes=model_bytes.getvalue())
