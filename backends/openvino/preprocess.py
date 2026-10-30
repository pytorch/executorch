# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import final, List

import torch
from executorch.backends.openvino._passes import (
    CastReductionInputPass,
    DecomposeFloorDividePass,
    FoldConstantCumsumPass,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass
from openvino.frontend.pytorch.torchdynamo.compile import (  # type: ignore[import-untyped]
    openvino_compile,
)

# The values of the inference_precision compile spec. The runtime
# (OpenvinoBackend.cpp) accepts the same ones.
_PRECISIONS = ("f32", "f16", "bf16")


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
        compile_options = {}
        for spec in module_compile_spec:
            compile_options[spec.key] = spec.value.decode()
        device = compile_options.get("device", "CPU")
        requested_precision = compile_options.get("inference_precision")
        # Some OpenVINO releases accept an unknown value. Checking before the
        # conversion fails fast and keeps it out of the program, where the
        # runtime would only fail at load.
        if requested_precision and requested_precision not in _PRECISIONS:
            raise ValueError(
                f"OpenVINO backend: unsupported inference_precision "
                f"'{requested_precision}', expected one of {', '.join(_PRECISIONS)}"
            )

        for pass_cls in [
            DimOrderOpsRevertPass,
            DecomposeFloorDividePass,
            CastReductionInputPass,
        ]:
            result = pass_cls()(edge_program.graph_module)
            if result and result.graph_module:
                edge_program._graph_module = result.graph_module

        input_names = edge_program.graph_signature.user_inputs
        args = []
        for node in edge_program.graph.nodes:
            if node.target in input_names:
                args.append(node.meta["val"])

        # The converter chooses some lowerings (e.g. boolean-mask indexing) from
        # the input types, so they must be known before conversion starts.
        module = edge_program.module()
        FoldConstantCumsumPass()(module)
        decoder = TorchFXPythonDecoder(
            module, input_shapes=input_shapes, input_types=input_types
        )
        frontend = FrontEndManager().load_by_framework("pytorch")
        ppp = PrePostProcessor(frontend.convert(frontend.load(decoder)))
        # The converter takes an int input, like a SymInt size, as a scalar, but
        # the runtime passes it as an int64 tensor of shape [1].
        for i in int_inputs:
            ppp.input(i).tensor().set_shape([1])
            ppp.input(i).preprocess().custom(lambda x: opset13.squeeze(x, 0).output(0))
        # On ARM CPUs OpenVINO fixes the precision here, not when the runtime
        # imports the model, so this must follow the same rule as the runtime.
        precision = requested_precision or ("f32" if device.startswith("CPU") else None)
        config = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        compiled = compile_model(ppp.build(), device, config)
        model_bytes = compiled.export_model()

        return PreprocessResult(processed_bytes=model_bytes.getvalue())
