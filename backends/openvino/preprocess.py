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
from openvino import compile_model, opset13, opset16  # type: ignore[import-untyped]
from openvino.frontend import FrontEndManager  # type: ignore[import-untyped]
from openvino.frontend.pytorch.fx_decoder import (  # type: ignore[import-untyped]
    TorchFXPythonDecoder,
)
from openvino.frontend.pytorch.utils import (  # type: ignore[import-untyped]
    pt_to_ov_type_map,
)
from openvino.preprocess import PrePostProcessor  # type: ignore[import-untyped]
from torch.fx.passes.shape_prop import _extract_tensor_metadata

# The values of the inference_precision compile spec. The runtime
# (OpenvinoBackend.cpp) accepts the same ones.
_PRECISIONS = ("f32", "f16", "bf16")


def _match_partition_inputs(
    model, module: torch.fx.GraphModule, input_shapes: list, input_types: list
) -> None:
    """Make the converted model take the inputs the runtime passes."""
    # The converter takes an unused first input for forward's self and removes
    # it, but the runtime passes every partition input. The cumsum fold can
    # leave the first input unused.
    parameters = list(model.get_parameters())
    placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
    if len(parameters) == len(input_types) - 1 and not placeholders[0].users:
        for parameter in parameters:
            model.remove_parameter(parameter)
        first = opset13.parameter(
            input_shapes[0], pt_to_ov_type_map[str(input_types[0])]
        )
        model.add_parameters([first, *parameters])
    # The runtime binds inputs by position and refuses a wrong count, so any
    # other mismatch fails here instead of on every call.
    if len(model.get_parameters()) != len(input_types):
        raise RuntimeError(
            f"OpenVINO backend: expected {len(input_types)} parameters in the "
            f"converted model, got {len(model.get_parameters())}"
        )


def _copy_returned_inputs(model) -> None:
    """Make the converted model compute each output that is one of its inputs.

    The converter turns copies like aten.clone into nothing, so such an output
    is the input itself. The CPU plugin does not write it, and can overwrite
    the input with the output buffer's old contents. OpenVINO keeps an
    Identity between an input and an output, and the CPU plugin runs it as a
    copy, while it can remove a Convert to the same type.
    """
    for result in model.get_results():
        source = result.input_value(0)
        if source.get_node().get_type_name() == "Parameter":
            result.input(0).replace_source_output(opset16.identity(source).output(0))


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

        module = edge_program.module()
        FoldConstantCumsumPass()(module)
        # The converter chooses some lowerings (e.g. boolean-mask indexing) from
        # the input types, so they must be known before conversion starts.
        # The converter reads ranks only from tensor_meta, which edge graphs
        # leave empty, and some ops need one, like aten.index after
        # aten.index_put.
        for node in module.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and not node.meta.get("tensor_meta"):
                node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        decoder = TorchFXPythonDecoder(
            module, input_shapes=input_shapes, input_types=input_types
        )
        frontend = FrontEndManager().load_by_framework("pytorch")
        model = frontend.convert(frontend.load(decoder))
        _match_partition_inputs(model, module, input_shapes, input_types)
        _copy_returned_inputs(model)
        ppp = PrePostProcessor(model)
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
