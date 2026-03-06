# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT backend implementation for ExecuTorch."""

import logging
from typing import Any, Dict, final, List, Optional, Tuple

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram

from executorch.backends.nvidia.tensorrt.compile_spec import (
    TensorRTCompileSpec,
    TensorRTPrecision,
)
from executorch.backends.nvidia.tensorrt.converter_registry import (
    lookup_converter,
    needs_edge_program,
)
from executorch.backends.nvidia.tensorrt.serialization import (
    serialize_blob,
    TensorRTBlobMetadata,
    TensorRTIOBinding,
)
from executorch.backends.nvidia.tensorrt.converters import (
    clear_converter_weight_storage,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@final
class TensorRTBackend(BackendDetails):
    """TensorRT backend for accelerating models on NVIDIA GPUs.

    This backend compiles ExecuTorch edge programs to TensorRT engines
    for optimized inference on NVIDIA hardware.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Compile edge program to TensorRT engine.

        Args:
            edge_program: The edge dialect program to compile.
            compile_specs: Backend-specific compilation options.

        Returns:
            PreprocessResult containing the serialized TensorRT engine.
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise RuntimeError(
                "TensorRT is not available. Please install TensorRT to use this backend."
            ) from e

        # Import converters to trigger registration
        from executorch.backends.nvidia.tensorrt import (  # noqa: F401
            converters as _converters,
        )
        from executorch.backends.nvidia.tensorrt.converter_utils import (
            ConversionContext,
            get_op_name,
            get_trt_tensor,
            torch_dtype_to_trt,
        )

        # Parse compile specs
        spec = TensorRTCompileSpec.from_compile_specs(compile_specs)
        if spec is None:
            spec = TensorRTCompileSpec()

        graph_module = edge_program.graph_module

        # Identify input and output nodes
        input_nodes = _get_input_nodes(graph_module, edge_program)
        output_nodes = _get_output_nodes(graph_module)

        # Create TensorRT builder and network
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        if network is None:
            raise RuntimeError("Failed to create TensorRT network")

        # Create conversion context for this build
        ctx = ConversionContext(net=network)

        # Build the network
        input_map = _add_network_inputs(network, input_nodes, torch_dtype_to_trt)
        # Add params/buffers as constant tensors
        _add_params_to_input_map(
            graph_module, edge_program, network, input_map, get_trt_tensor
        )
        _process_graph_nodes(
            graph_module, edge_program, network, input_map, get_trt_tensor, get_op_name, ctx
        )
        _mark_network_outputs(network, output_nodes, input_map)

        # Collect I/O bindings from network
        io_bindings = _collect_io_bindings(network)

        # Configure and build engine
        config = _create_builder_config(builder, spec, trt)
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Serialize with metadata
        metadata = TensorRTBlobMetadata(io_bindings=io_bindings)
        blob = serialize_blob(bytes(serialized_engine), metadata)

        return PreprocessResult(processed_bytes=blob)


def _get_input_nodes(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
) -> List[torch.fx.Node]:
    """Get graph input placeholder nodes (excluding parameters/buffers)."""
    input_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if not _is_param_or_buffer(node, exported_program):
                input_nodes.append(node)
    return input_nodes


def _get_output_nodes(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    """Get nodes that are graph outputs."""
    output_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for arg in node.args:
                if isinstance(arg, (list, tuple)):
                    output_nodes.extend(
                        item for item in arg if isinstance(item, torch.fx.Node)
                    )
                elif isinstance(arg, torch.fx.Node):
                    output_nodes.append(arg)
    return output_nodes


def _is_param_or_buffer(
    node: torch.fx.Node, exported_program: ExportedProgram
) -> bool:
    """Check if a placeholder node is a parameter or buffer."""
    if node.op != "placeholder":
        return False

    if hasattr(exported_program, "state_dict"):
        if node.name in exported_program.state_dict:
            return True

    if hasattr(exported_program, "graph_signature"):
        sig = exported_program.graph_signature
        if hasattr(sig, "inputs_to_parameters"):
            if node.name in sig.inputs_to_parameters:
                return True
        if hasattr(sig, "inputs_to_buffers"):
            if node.name in sig.inputs_to_buffers:
                return True

    return False


def _add_params_to_input_map(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    get_trt_tensor_fn: Any,
) -> None:
    """Add parameters and buffers as constant TensorRT tensors to input_map.

    In ExecuTorch's edge dialect, parameters are often "lifted" as placeholder
    inputs rather than get_attr nodes. This function identifies these placeholder
    nodes that represent parameters/buffers and adds them to input_map as
    TensorRT constant tensors.
    """
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            # Skip if already in input_map (it's a real input, not a param)
            if node in input_map:
                continue

            param_tensor = None

            # Try to get from state_dict first
            if hasattr(exported_program, "state_dict"):
                if node.name in exported_program.state_dict:
                    param_tensor = exported_program.state_dict[node.name]

            # Try to get from graph_signature mapping
            if param_tensor is None and hasattr(exported_program, "graph_signature"):
                sig = exported_program.graph_signature
                param_name = None
                if hasattr(sig, "inputs_to_parameters"):
                    param_name = sig.inputs_to_parameters.get(node.name)
                if param_name is None and hasattr(sig, "inputs_to_buffers"):
                    param_name = sig.inputs_to_buffers.get(node.name)

                if param_name is not None and hasattr(exported_program, "state_dict"):
                    param_tensor = exported_program.state_dict.get(param_name)

            # If we found a parameter tensor, add it to input_map
            if param_tensor is not None:
                if isinstance(param_tensor, torch.nn.Parameter):
                    param_tensor = param_tensor.data
                if isinstance(param_tensor, torch.Tensor):
                    # Convert int64/int32 tensors to float32 for TensorRT compatibility
                    # These are often used in elementwise operations with float tensors
                    # (e.g., batch norm statistics in MobileNetV3)
                    original_dtype = param_tensor.dtype
                    if param_tensor.dtype in (torch.int32, torch.int64):
                        param_tensor = param_tensor.float()
                        logger.debug(
                            f"Converting param {node.name} from {original_dtype} to float32 "
                            f"for TensorRT compatibility"
                        )
                    elif param_tensor.dtype == torch.float64:
                        param_tensor = param_tensor.float()
                    input_map[node] = get_trt_tensor_fn(
                        network, param_tensor, f"param_{node.name}"
                    )


def _get_tensor_shape_and_dtype(
    node: torch.fx.Node,
) -> Tuple[Optional[Tuple[int, ...]], Optional[torch.dtype]]:
    """Extract tensor shape and dtype from node metadata."""
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return tuple(val.shape), val.dtype
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            return tuple(val.shape), val.dtype
    return None, None


def _get_attr_value(
    graph_module: torch.fx.GraphModule, attr_name: str
) -> Optional[torch.Tensor]:
    """Get attribute value from graph module."""
    try:
        parts = attr_name.split(".")
        obj = graph_module
        for part in parts:
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, torch.nn.Parameter):
            return obj.data
        return None
    except AttributeError:
        return None


def _add_network_inputs(
    network: Any,
    input_nodes: List[torch.fx.Node],
    dtype_converter: Any,
) -> Dict[torch.fx.Node, Any]:
    """Add input tensors to TensorRT network."""
    input_map: Dict[torch.fx.Node, Any] = {}

    for input_node in input_nodes:
        shape, dtype = _get_tensor_shape_and_dtype(input_node)
        if shape is None:
            raise RuntimeError(
                f"Cannot determine shape for input node: {input_node.name}"
            )

        trt_dtype = dtype_converter(dtype if dtype else torch.float32)
        trt_input = network.add_input(
            name=input_node.name,
            dtype=trt_dtype,
            shape=shape,
        )
        if trt_input is None:
            raise RuntimeError(f"Failed to add input to network: {input_node.name}")

        input_map[input_node] = trt_input

    return input_map


def _process_graph_nodes(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    get_trt_tensor_fn: Any,
    get_op_name_fn: Any,
    ctx: Any = None,
) -> None:
    """Process graph nodes and convert to TensorRT layers.

    Args:
        graph_module: The FX graph module to process.
        exported_program: The ExportedProgram for weight extraction.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        get_trt_tensor_fn: Function to create TensorRT constant tensors.
        get_op_name_fn: Function to extract operation name from nodes.
        ctx: Optional ConversionContext for unique layer naming.
    """
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            op_name = get_op_name_fn(node)

            converter = lookup_converter(op_name)
            if converter is None:
                raise RuntimeError(f"No converter registered for operation: {op_name}")

            # Check if converter needs edge_program for weight extraction
            if needs_edge_program(op_name):
                output_tensor = converter(node, network, input_map, exported_program, ctx)
            else:
                output_tensor = converter(node, network, input_map, ctx)

            input_map[node] = output_tensor

        elif node.op == "get_attr":
            attr_name = node.target
            param = _get_attr_value(graph_module, attr_name)
            if param is not None:
                input_map[node] = get_trt_tensor_fn(
                    network, param, f"param_{node.name}"
                )


def _mark_network_outputs(
    network: Any,
    output_nodes: List[torch.fx.Node],
    input_map: Dict[torch.fx.Node, Any],
) -> None:
    """Mark network outputs in TensorRT network."""
    for output_node in output_nodes:
        if output_node not in input_map:
            raise RuntimeError(
                f"Output node not found in input_map: {output_node.name}"
            )

        output_tensor = input_map[output_node]
        if hasattr(output_tensor, "name"):
            output_tensor.name = f"output_{output_node.name}"
        network.mark_output(output_tensor)


def _trt_dtype_to_string(dtype: Any) -> str:
    """Convert TensorRT DataType to string representation."""
    dtype_name = str(dtype)
    # dtype looks like "DataType.FLOAT" or "DataType.HALF"
    if "." in dtype_name:
        dtype_name = dtype_name.split(".")[-1]

    dtype_map = {
        "FLOAT": "float32",
        "HALF": "float16",
        "INT8": "int8",
        "INT32": "int32",
        "INT64": "int64",
        "BOOL": "bool",
        "UINT8": "uint8",
        "FP8": "float8",
        "BF16": "bfloat16",
    }
    return dtype_map.get(dtype_name, "float32")


def _collect_io_bindings(network: Any) -> List[TensorRTIOBinding]:
    """Collect I/O binding information from TensorRT network.

    Args:
        network: TensorRT network definition.

    Returns:
        List of TensorRTIOBinding with input/output tensor metadata.
    """
    bindings = []

    # Collect inputs
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        bindings.append(
            TensorRTIOBinding(
                name=tensor.name,
                dtype=_trt_dtype_to_string(tensor.dtype),
                shape=list(tensor.shape),
                is_input=True,
            )
        )

    # Collect outputs
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        bindings.append(
            TensorRTIOBinding(
                name=tensor.name,
                dtype=_trt_dtype_to_string(tensor.dtype),
                shape=list(tensor.shape),
                is_input=False,
            )
        )

    return bindings


def _create_builder_config(builder: Any, spec: TensorRTCompileSpec, trt: Any) -> Any:
    """Create and configure TensorRT builder config."""
    config = builder.create_builder_config()
    if config is None:
        raise RuntimeError("Failed to create TensorRT builder config")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, spec.workspace_size)

    # Disable TF32 for strict FP32 precision on Ampere+ GPUs.
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    # Report build progress if TRT supports IProgressMonitor.
    if hasattr(trt, "IProgressMonitor"):

        class _ProgressMonitor(trt.IProgressMonitor):
            def __init__(self):
                self._seen = set()
    # Report build progress if TRT supports IProgressMonitor.
    if hasattr(trt, "IProgressMonitor"):

        class _ProgressMonitor(trt.IProgressMonitor):
            def __init__(self):
                trt.IProgressMonitor.__init__(self)
                self._seen = set()

            def phase_start(self, phase_name, parent_phase, num_steps):
                key = (phase_name, parent_phase)
                if key not in self._seen:
                    self._seen.add(key)
                    indent = "    " if parent_phase else "  "
                    print(f"{indent}TRT: {phase_name}", flush=True)

            def step_complete(self, phase_name, step):
                return True

            def phase_finish(self, phase_name):
                pass

        config.progress_monitor = _ProgressMonitor()

    # TensorRT 10.6+ enables WEIGHT_STREAMING by default, which generates
    # weight-separated plan files that require IStreamReader for deserialization.
    # We disable this flag to generate standard plan files that can be
    # deserialized with the simpler deserializeCudaEngine(data, size) API.
    if hasattr(trt.BuilderFlag, "WEIGHT_STREAMING"):
        config.clear_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    if spec.precision == TensorRTPrecision.FP16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 not supported on this platform, using FP32")

    if spec.precision == TensorRTPrecision.INT8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            logger.warning("INT8 not supported on this platform, using FP32")

    if spec.strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    if spec.dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = spec.dla_core
        if spec.allow_gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    return config
