# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import final, List, Optional

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
from executorch.exir.tensor import dim_order_from_stride
from openvino import compile_model, opset13, opset16  # type: ignore[import-untyped]
from openvino.frontend import FrontEndManager  # type: ignore[import-untyped]
from openvino.frontend.pytorch.fx_decoder import (  # type: ignore[import-untyped]
    TorchFXPythonDecoder,
)
from openvino.frontend.pytorch.utils import (  # type: ignore[import-untyped]
    pt_to_ov_type_map,
)
from openvino.preprocess import PrePostProcessor  # type: ignore[import-untyped]
from torch.export.graph_signature import OutputKind
from torch.fx.passes.shape_prop import _extract_tensor_metadata

# The values of the inference_precision compile spec. The runtime
# (OpenvinoBackend.cpp) accepts the same ones.
_PRECISIONS = ("f32", "f16", "bf16")

# The runtime passes the tensors of a blob with this header with their sizes in
# memory order, and those of an older blob, without it, as before. The header
# holds the magic and its own length, padded to 16 bytes.
_HEADER_MAGIC = b"ETOVHDR1"
_HEADER_LENGTH = 16
_HEADER = (_HEADER_MAGIC + _HEADER_LENGTH.to_bytes(4, "little")).ljust(
    _HEADER_LENGTH, b"\0"
)


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


def _inference_precision(device: str, requested: Optional[str]) -> Optional[str]:
    """The INFERENCE_PRECISION_HINT to pass, or None for the device's default."""
    return requested or ("f32" if device.startswith("CPU") else None)


def _copy_returned_inputs(model, device: str) -> None:
    """Make the converted model compute each output that is one of its inputs.

    The converter turns copies like aten.clone into nothing, so such an output
    is the input itself. The CPU plugin does not write it, and can overwrite
    the input with the output buffer's old contents. OpenVINO keeps an
    Identity between an input and an output, and the CPU plugin runs it as a
    copy, while it can remove a Convert to the same type. On an ARM CPU with
    float16 support, at f16 or bf16, it rounds this copy of a float32 input,
    like its other float32 outputs.
    Other devices are left as they are, since neither the problem nor Identity
    was checked there.
    """
    if not device.startswith("CPU"):
        return
    for result in model.get_results():
        source = result.input_value(0)
        if source.get_node().get_type_name() == "Parameter":
            result.input(0).replace_source_output(opset16.identity(source).output(0))


def _non_default_dim_order(val) -> Optional[tuple]:
    """The tensor's dim order as ExecuTorch stores it, if it is not the
    default."""
    if not isinstance(val, torch.Tensor):
        return None
    order = dim_order_from_stride(val.stride(), tuple(val.shape))
    return order if order != tuple(range(val.dim())) else None


def _partition_inputs(edge_program: ExportedProgram):
    """The type and shape of each partition input, in the order the runtime
    passes them, the positions of the int inputs, and the dim order of each
    tensor input that is not in the default one, by position."""
    input_names = edge_program.graph_signature.user_inputs
    input_types: list[torch.dtype] = []
    input_shapes = []
    int_inputs = []
    input_dim_orders: dict[int, tuple] = {}
    for node in edge_program.graph.nodes:
        if node.target not in input_names:
            continue
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            if order := _non_default_dim_order(val):
                input_dim_orders[len(input_types)] = order
            input_types.append(val.dtype)
            input_shapes.append([d if isinstance(d, int) else -1 for d in val.shape])
        elif isinstance(val, (int, torch.SymInt)) and not isinstance(val, bool):
            int_inputs.append(len(input_types))
            input_types.append(torch.int64)
            input_shapes.append([])
        else:
            raise RuntimeError(
                f"OpenVINO backend: partition input {node.target} is a "
                f"{type(val).__name__}; only tensors and ints are supported"
            )
    return input_types, input_shapes, int_inputs, input_dim_orders


def _output_dim_orders(edge_program: ExportedProgram) -> dict[int, tuple]:
    """The dim order of each partition output that is not in the default one,
    by its position among the outputs of edge_program.module(), which returns
    only the user outputs."""
    outputs = [
        node
        for node, spec in zip(
            edge_program.graph.output_node().args[0],
            edge_program.graph_signature.output_specs,
        )
        if spec.kind == OutputKind.USER_OUTPUT
    ]
    return {
        i: order
        for i, node in enumerate(outputs)
        if isinstance(node, torch.fx.Node)
        and (order := _non_default_dim_order(node.meta.get("val")))
    }


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

        input_types, input_shapes, int_inputs, input_dim_orders = _partition_inputs(
            edge_program
        )
        output_dim_orders = _output_dim_orders(edge_program)

        module = edge_program.module()
        FoldConstantCumsumPass()(module)
        # The converter reads ranks only from tensor_meta, which edge graphs
        # leave empty, and some ops need one, like aten.index after
        # aten.index_put.
        for node in module.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and not node.meta.get("tensor_meta"):
                node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        # The converter chooses some lowerings (e.g. boolean-mask indexing) from
        # the input types, so they must be known before conversion starts.
        decoder = TorchFXPythonDecoder(
            module, input_shapes=input_shapes, input_types=input_types
        )
        frontend = FrontEndManager().load_by_framework("pytorch")
        model = frontend.convert(frontend.load(decoder))
        _match_partition_inputs(model, module, input_shapes, input_types)
        _copy_returned_inputs(model, device)
        ppp = PrePostProcessor(model)
        # The converter takes an int input, like a SymInt size, as a scalar, but
        # the runtime passes it as an int64 tensor of shape [1].
        for i in int_inputs:
            ppp.input(i).tensor().set_shape([1])
            ppp.input(i).preprocess().custom(lambda x: opset13.squeeze(x, 0).output(0))
        # OpenVINO computes in the default dim order, so a partition input or
        # output in another one gets a transpose, and the runtime passes its
        # bytes with the sizes in memory order.
        for i, dim_order in input_dim_orders.items():
            inverse = [dim_order.index(d) for d in range(len(dim_order))]
            ppp.input(i).preprocess().convert_layout(inverse)
        for i, dim_order in output_dim_orders.items():
            ppp.output(i).postprocess().convert_layout(list(dim_order))
        # On ARM CPUs OpenVINO fixes the precision here, not when the runtime
        # imports the model, so this must follow the same rule as the runtime.
        precision = _inference_precision(device, requested_precision)
        config = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        compiled = compile_model(ppp.build(), device, config)
        model_bytes = compiled.export_model()

        return PreprocessResult(processed_bytes=_HEADER + model_bytes.getvalue())
