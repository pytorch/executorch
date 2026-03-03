from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tensorrt as trt
import torch
import torch_tensorrt
from torch.nn import Module
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform, dtype
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import cls_supports_debugger
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM
from torch_tensorrt.logging import TRT_LOGGER
from torch_tensorrt.runtime._utils import (
    _is_switch_required,
    _select_rt_device,
    multi_gpu_device_check,
)

logger = logging.getLogger(__name__)


class DynamicOutputAllocator(trt.IOutputAllocator):  # type: ignore[misc]
    def __init__(self, output_dtypes: Dict[str, torch.dtype]) -> None:
        trt.IOutputAllocator.__init__(self)
        self.buffers: Dict[str, torch.Tensor] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.dtypes: Dict[str, torch.dtype] = output_dtypes

    def reallocate_output_async(
        self,
        tensor_name: str,
        memory: int,
        size: int,
        alignment: int,
        stream: torch.cuda.Stream,
    ) -> Any:
        shape = (size,)
        if tensor_name not in self.buffers:
            self.buffers[tensor_name] = torch.empty(
                shape,
                dtype=self.dtypes[tensor_name],
                device=torch.cuda.current_device(),
            )
        else:
            if self.buffers[tensor_name].shape != shape:
                self.buffers[tensor_name] = torch.empty(
                    shape,
                    dtype=self.dtypes[tensor_name],
                    device=torch.cuda.current_device(),
                )
        return self.buffers[tensor_name].data_ptr()

    def notify_shape(self, tensor_name: str, shape: Tuple[int, ...]) -> None:
        self.shapes[tensor_name] = tuple(shape)


class TorchTRTRuntimeStates:
    def __init__(self, new_cudagraphs: bool):
        # Indicates whether CUDAGraphs were enabled in the previous execute_engine
        self.old_cudagraphs = new_cudagraphs
        # Indicates whether pre-allocated output was enabled in the previous execute_engine
        self.old_pre_allocated_outputs = False
        # Indicates whether context has changed
        self.context_changed = False

    def set_runtime_states(
        self,
        new_cudagraphs: bool,
        new_pre_allocated_output: bool,
        shape_changed: bool,
    ) -> Tuple[bool, bool, bool]:
        # Evaluates whether certain conditions are met to enable CUDA Graph recording or to use pre-allocated outputs
        # based on the current and previous states, as well as input shape has changed
        need_cudagraphs_record = False
        can_use_pre_allocated_outputs = False
        need_cudagraphs_reset = False

        # CUDA Graph recording is needed if CUDA graphs is enabled and:
        # - CUDA graphs were previously disabled
        # - or the shape has changed
        # - or the execution context has changed (e.g., weight streaming)
        if new_cudagraphs and (
            not self.old_cudagraphs or shape_changed or self.context_changed
        ):
            need_cudagraphs_record = True

        # Pre-allocated output can be used when previous and current state are true without shape change
        if (
            self.old_pre_allocated_outputs
            and new_pre_allocated_output
            and (not shape_changed)
        ):
            can_use_pre_allocated_outputs = True

        if not new_cudagraphs or shape_changed or self.context_changed:
            need_cudagraphs_reset = True

        self.old_cudagraphs = new_cudagraphs
        self.old_pre_allocated_outputs = new_pre_allocated_output
        # reset flag
        self.context_changed = False

        return (
            need_cudagraphs_record,
            can_use_pre_allocated_outputs,
            need_cudagraphs_reset,
        )


@cls_supports_debugger
class PythonTorchTensorRTModule(Module):  # type: ignore[misc]
    """PythonTorchTensorRTModule is a PyTorch module which encompasses an arbitrary TensorRT Engine.

    This module is backed by the Torch-TensorRT runtime and is only compatible with
    FX / Dynamo / Python deployments. This module cannot be serialized to torchscript via torch.jit.trace for C++ deployment.
    """

    def __init__(
        self,
        serialized_engine: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),
        weight_name_map: Optional[dict[Any, Any]] = None,
        requires_output_allocator: bool = False,
        _debugger_config: Optional[DebuggerConfig] = None,
    ):
        """Takes a name, target device, serialized TensorRT engine, and binding names / order and constructs
        a PyTorch ``torch.nn.Module`` around it. Uses TensorRT Python APIs to run the engine

        Arguments:
            serialized_engine (bytes): Serialized TensorRT engine in the form of a bytearray
            input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
            output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned

        Keyword Arguments:
            name (str): Name for module
            settings (torch_tensorrt.dynamo.CompilationSettings): Settings used to compile engine, assumes engine was built with default compilation settings if object not passed
            weight_name_map (dict): Mapping of engine weight name to state_dict weight name
            requires_output_allocator (bool): Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)

        Example:

            .. code-block:: py

                trt_module = PythonTorchTensorRTModule(
                    engine_str,
                    input_binding_names=["x"],
                    output_binding_names=["output"],
                    name="my_module",
                    settings=CompilationSettings(device=torch.cuda.current_device)
                )

        """
        self.context: Any
        self._debugger_config: Optional[DebuggerConfig] = _debugger_config
        super(PythonTorchTensorRTModule, self).__init__()
        self._register_state_dict_hook(PythonTorchTensorRTModule._on_state_dict)

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()

        self.name = name
        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None

        # TODO: Make the below a Dictionary {shape: cudagraph}
        self.shape_key: Optional[str] = None

        # See https://github.com/pytorch/pytorch/blob/acfe237a71af609e837a34bb38048aa8acb8eb4d/torch/cuda/graphs.py#L92-L98
        # Unused currently - to be used by Dynamic Shape support implementation
        self.memory_pool = None

        self.serialized_engine = serialized_engine
        self.input_names = (
            input_binding_names if input_binding_names is not None else []
        )
        self.output_names = (
            output_binding_names if output_binding_names is not None else []
        )
        self.initialized = False
        self.target_device_id = (
            settings.device.gpu_id
            if settings.device is not None
            else Device._current_device().gpu_id
        )
        self.target_device_properties = torch.cuda.get_device_properties(
            self.target_device_id
        )
        self.profiling_enabled = (
            _debugger_config.save_engine_profile
            if _debugger_config is not None
            else False
        )
        self.settings = settings
        self.engine = None
        self.weight_name_map = weight_name_map
        self.target_platform = Platform.current_platform()
        self.runtime_states = TorchTRTRuntimeStates(
            torch_tensorrt.runtime.get_cudagraphs_mode()
        )

        self.cudagraphs_enabled = False
        self.pre_allocated_outputs: List[torch.Tensor] = []
        self.use_pre_allocated_outputs = False

        self.requires_output_allocator = requires_output_allocator
        self.output_allocator: Optional[DynamicOutputAllocator] = None
        self.use_output_allocator_outputs = False
        self.device = torch.cuda.current_device()
        self.cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()
        # If the output tensor is not owned by the engine (output_tensors_are_unowned=True), we need to create a new output tensor in each forward pass
        self.output_tensors_are_unowned = False
        if self.serialized_engine is not None and not self.settings.lazy_engine_init:
            self.setup_engine()

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        """
        Flag to set if the output tensors of this engine are solely owned by the Torch-TensorRT Runtime or if they might be shared with a user.
        If the tensors are not owned by the runtime, then they must be recreated on every forward call which may have implications for performance.
        Typically only the final engine in a graph requires output tensors to be unowned and there are performance gains to be had for intermediate engines to manage their own standing memory.
        Therefore this should only be set to True for the final module in a graph and leave false for intermediate modules.

        Args:
            enabled: bool
                Whether to set the flag to True.

        """
        self.output_tensors_are_unowned = enabled

    def get_streamable_device_memory_budget(self) -> Any:
        return self.engine.streamable_weights_size

    def get_automatic_device_memory_budget(self) -> Any:
        return self.engine.get_weight_streaming_automatic_budget()

    def get_device_memory_budget(self) -> Any:
        return self.engine.weight_streaming_budget_v2

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        # Recreating the context because weight streaming budget cannot be modified while there are active context.
        if self.context is not None:
            del self.context
        budget_bytes = self._set_device_memory_budget(budget_bytes)
        self.context = self.engine.create_execution_context()
        self.runtime_states.context_changed = True
        return budget_bytes

    def _set_device_memory_budget(self, budget_bytes: int) -> int:
        # Disable weight streaming for invalid budget size
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        self.engine.weight_streaming_budget_v2 = budget_bytes
        if self.engine.weight_streaming_budget_v2 != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = self.engine.weight_streaming_budget_v2
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")

        return budget_bytes

    def set_default_device_memory_budget(self) -> int:
        budget_bytes = self.get_automatic_device_memory_budget()
        # Set automatic weight streaming budget as default when context is created
        logger.debug(f"Weight streaming budget set to {budget_bytes}B")
        return self._set_device_memory_budget(budget_bytes)

    def setup_engine(self) -> None:
        assert (
            self.target_platform == Platform.current_platform()
        ), f"TensorRT engine was not built to target current platform (target: {self.target_platform}, current: {Platform.current_platform()})"

        self.initialized = True
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)
        if self.settings.enable_weight_streaming:
            self.set_default_device_memory_budget()
        self.context = self.engine.create_execution_context()
        assert self.context is not None, "Failed to create execution context"
        assert self.engine.num_io_tensors == (
            len(self.input_names) + len(self.output_names)
        )

        self.input_dtypes = [
            dtype._from(self.engine.get_tensor_dtype(input_name))
            for input_name in self.input_names
        ]

        self.input_shapes = [
            self.engine.get_tensor_shape(input_name) for input_name in self.input_names
        ]
        self.output_dtypes = [
            dtype._from(self.engine.get_tensor_dtype(output_name)).to(torch.dtype)
            for output_name in self.output_names
        ]
        self.output_shapes = [
            self.engine.get_tensor_shape(output_name)
            for output_name in self.output_names
        ]

        if self.requires_output_allocator:
            self.create_output_allocator()

        if torch_tensorrt.runtime.get_cudagraphs_mode():
            self.cudagraph = torch.cuda.CUDAGraph()

        self.is_shape_inference_io = {
            input_name: self.engine.is_shape_inference_io(input_name)
            for input_name in self.input_names
        }

    def _check_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("PythonTorchTensorRTModule is not initialized.")

    def _on_state_dict(self, state_dict: Dict[str, Any], prefix: str, _: Any) -> None:
        state_dict[prefix + "engine"] = self.serialized_engine
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "platform"] = self.target_platform

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: Any,
        missing_keys: Any,
        unexpected_keys: Any,
        error_msgs: Any,
    ) -> None:
        self.serialized_engine = state_dict[prefix + "engine"]
        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]
        self.target_platform = state_dict[prefix + "platform"]

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()
        self.setup_engine()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("engine", None)
        state.pop("context", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.setup_engine()

    def __deepcopy__(self, memo: Any) -> PythonTorchTensorRTModule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__setstate__(self.__getstate__())
        return result

    def _reset_captured_graph(self) -> None:
        if self.cudagraph:
            self.cudagraph.reset()
            self.cudagraph = None

    def __del__(self) -> None:
        self._reset_captured_graph()

    def setup_input_tensors(
        self,
        contiguous_inputs: List[torch.Tensor],
        cudagraphs_enabled: bool,
        need_cudagraphs_record: bool,
    ) -> None:
        for i, input_name in enumerate(self.input_names):
            if not contiguous_inputs[i].is_cuda:
                logger.warning(
                    f"Detected input {input_name} of engine {self.engine.name} is not on a cuda device. "
                    "This tensor is being moved by the runtime but for performance considerations, "
                    "ensure your inputs are all on GPU and open an issue here "
                    "(https://github.com/pytorch/TensorRT/issues) if this warning persists."
                )
                contiguous_inputs = (
                    contiguous_inputs[:i]
                    + [contiguous_inputs[i].cuda()]
                    + contiguous_inputs[i + 1 :]
                )

            assert (
                contiguous_inputs[i].dtype == self.input_dtypes[i]
            ), f"Dtype mismatch for {i}th input({input_name}). Expect {self.input_dtypes[i]}, got {contiguous_inputs[i].dtype}."

            if need_cudagraphs_record:
                # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                # Clone is required to avoid re-using user-provided GPU memory
                self._input_buffers[i] = contiguous_inputs[i].clone()

            # For shape tensors, we use CPU pointers and for data tensors, we use GPU pointers
            # as per TensorRT requirements
            if self.is_shape_inference_io[input_name]:
                # Shape tensor inputs are casted to int64 explicitly
                # Currently Torch CPU pointers are not working; numpy pointers are used instead
                # to refer to underlying memory
                inputs_cpu = contiguous_inputs[i].cpu().to(torch.int64).numpy().copy()
                self.context.set_tensor_address(input_name, inputs_cpu.ctypes.data)
            else:
                self.context.set_input_shape(
                    input_name, tuple(contiguous_inputs[i].shape)
                )
                if cudagraphs_enabled:
                    self._input_buffers[i].copy_(contiguous_inputs[i])
                    self.context.set_tensor_address(
                        input_name, self._input_buffers[i].data_ptr()
                    )
                else:
                    self.context.set_tensor_address(
                        input_name, contiguous_inputs[i].data_ptr()
                    )

    def create_output_tensors(self) -> List[torch.Tensor]:
        # create output tensors
        outputs: List[torch.Tensor] = []

        for o, _ in enumerate(self.output_names):
            output = torch.empty(
                size=self.output_shapes[o],
                dtype=self.output_dtypes[o],
                device=self.device,
            )
            outputs.append(output)
        return outputs

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self.use_pre_allocated_outputs = enable

    def set_use_output_allocator(self, enable: bool) -> None:
        self.use_output_allocator_outputs = enable

    def create_output_allocator(self) -> None:
        if self.output_allocator is None:
            output_dtypes_dict = {}
            for o, output_name in enumerate(self.output_names):
                output_dtypes_dict[output_name] = self.output_dtypes[o]
            self.output_allocator = DynamicOutputAllocator(output_dtypes_dict)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:

        def run_standard_execution() -> torch.Tensor | Tuple[torch.Tensor, ...]:
            shape_changed = self.validate_input_shapes(contiguous_inputs)
            (
                need_cudagraphs_record,
                can_use_pre_allocated_outputs,
                need_cudagraphs_reset,
            ) = self.runtime_states.set_runtime_states(
                self.cudagraphs_enabled, self.use_pre_allocated_outputs, shape_changed
            )

            if need_cudagraphs_reset:
                self._reset_captured_graph()

            if need_cudagraphs_record:
                self._input_buffers = [None] * len(self.input_names)
                self._output_buffers = [None] * len(self.output_names)

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessInputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                assert len(contiguous_inputs) == len(
                    self.input_names
                ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(contiguous_inputs)}."

                self.setup_input_tensors(
                    contiguous_inputs, self.cudagraphs_enabled, need_cudagraphs_record
                )

                if shape_changed:
                    # Check if input shapes can be inferred.
                    uninferred_input_names = self.context.infer_shapes()
                    if uninferred_input_names:
                        logger.warning(
                            f"The shapes of the inputs: {uninferred_input_names} cannot be inferred and could lead to undefined behavior. \
                                    This could happen if the input tensor addresses/shapes haven't been configured correctly"
                        )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessOutputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                if can_use_pre_allocated_outputs:
                    outputs = self.pre_allocated_outputs
                else:
                    self.output_shapes = [
                        tuple(self.context.get_tensor_shape(output_name))
                        for output_name in self.output_names
                    ]
                    if DYNAMIC_DIM in self.output_shapes:
                        raise ValueError(
                            "Encountered dynamic output shapes during runtime. This could mean the network has data-dependent output shapes which is not currently supported."
                        )
                    outputs = self.create_output_tensors()

                for o, output_name in enumerate(self.output_names):
                    if need_cudagraphs_record:
                        self._output_buffers[o] = outputs[o].clone()

                    if self.cudagraphs_enabled:
                        self.context.set_tensor_address(
                            output_name, self._output_buffers[o].data_ptr()
                        )
                    else:
                        self.context.set_tensor_address(
                            output_name, outputs[o].data_ptr()
                        )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:TensorRTRuntime"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                self._caller_stream = torch.cuda.current_stream()
                if (
                    self._engine_stream == torch.cuda.default_stream()
                    or self._engine_stream is None
                ):
                    self._engine_stream = torch.cuda.Stream()

                self._engine_stream.wait_stream(self._caller_stream)

                with torch.cuda.stream(self._engine_stream):
                    if self.cudagraphs_enabled:
                        if need_cudagraphs_record:
                            self.cudagraph = torch.cuda.CUDAGraph()

                            if self.profiling_enabled:
                                self.cudagraph.enable_debug_mode()

                            with torch.cuda.graph(
                                self.cudagraph, stream=self._engine_stream
                            ):
                                self.context.execute_async_v3(
                                    self._engine_stream.cuda_stream
                                )

                            if self.profiling_enabled:
                                self.cudagraph.debug_dump(
                                    f"{DEBUG_LOGGING_DIR}/{self.name}_cudagraph.dot"
                                )

                        self.cudagraph.replay()  # type: ignore

                    else:
                        self.context.execute_async_v3(self._engine_stream.cuda_stream)

                self._caller_stream.wait_stream(self._engine_stream)

            # When the pre-allocated output mode is turned on, for intermediate modules, we only create the output in the first execution or when shape is changed.
            if self.use_pre_allocated_outputs and (
                self.output_tensors_are_unowned
                or not self.pre_allocated_outputs
                or shape_changed
            ):
                self.pre_allocated_outputs = self.create_output_tensors()

            if self.cudagraphs_enabled:
                for idx, o in enumerate(outputs):
                    o.copy_(self._output_buffers[idx])

            if len(outputs) == 1:
                return outputs[0]

            return outputs

        def run_output_allocator() -> torch.Tensor | Tuple[torch.Tensor, ...]:
            assert (
                not torch_tensorrt.runtime.get_cudagraphs_mode()
            ), "CUDA Graphs are not compatible with OutputAllocator."
            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessInputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                assert len(contiguous_inputs) == len(
                    self.input_names
                ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(contiguous_inputs)}."

                self.setup_input_tensors(contiguous_inputs, False, False)

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:SetupOutputAllocator"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                self.create_output_allocator()
                # need to set output allocator every run
                for output_name in self.output_names:
                    if not self.context.set_output_allocator(
                        output_name, self.output_allocator
                    ):
                        raise RuntimeError(
                            f"Failed to set output allocator for {output_name}"
                        )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:TensorRTRuntime"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                self._caller_stream = torch.cuda.current_stream()
                if (
                    self._engine_stream == torch.cuda.default_stream()
                    or self._engine_stream is None
                ):
                    self._engine_stream = torch.cuda.Stream()

                self._engine_stream.wait_stream(self._caller_stream)

                with torch.cuda.stream(self._engine_stream):
                    self.context.execute_async_v3(
                        self._engine_stream.cuda_stream
                    )  # The OutputAllocator is called by execute_async_v3()

                self._caller_stream.wait_stream(self._engine_stream)

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessOutputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                outputs = []
                assert self.output_allocator is not None
                for o, output_name in enumerate(self.output_names):
                    shape = self.output_allocator.shapes.get(output_name, None)
                    dtype = self.output_dtypes[o]
                    output = (
                        self.output_allocator.buffers.get(output_name, None)
                        .clone()
                        .detach()
                    )
                    prod = int(torch.prod(torch.tensor(shape)))
                    # When using the OutputAllocator, the allocated buffer might be larger than the size of the output,
                    # so we need to reshape the buffer to the output shape
                    output = output.reshape(-1).view(dtype)[:prod].reshape(shape)
                    outputs.append(output)

            if len(outputs) == 1:
                return outputs[0]

            return outputs

        self.cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()

        # Run forward function
        contiguous_inputs: List[torch.Tensor] = [
            (i.contiguous() if isinstance(i, torch.Tensor) else torch.tensor(i).cuda())
            for i in inputs
        ]
        with (
            torch.autograd.profiler.record_function("PythonTorchTensorRTModule:Forward")
            if self.profiling_enabled
            else nullcontext()
        ):
            self._check_initialized()

            # If in safe mode, check at each iteration for whether a switch is required
            if (
                torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE
            ):
                curr_device_id = torch.cuda.current_device()
                curr_device_properties = torch.cuda.get_device_properties(
                    curr_device_id
                )
                logger.debug(f"Current Device: cuda:{curr_device_id}")

                # If a switch is required, move all inputs to new device and set as active device
                if _is_switch_required(
                    curr_device_id,
                    self.target_device_id,
                    curr_device_properties,
                    self.target_device_properties,
                ):
                    device_id, _ = _select_rt_device(
                        curr_device_id,
                        self.target_device_id,
                        self.target_device_properties,
                    )

                    # Update current device
                    device = torch.device(device_id)
                    torch.cuda.set_device(device_id)

                    contiguous_inputs = [
                        tensor.to(device) for tensor in contiguous_inputs
                    ]
                    logger.warning(f"Moved all input Tensors to cuda:{device_id}")

            if self.requires_output_allocator:  # engine requires OA
                if self.cudagraphs_enabled:
                    raise RuntimeError(
                        "The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs."
                    )
                logger.debug("Using the dynamic allocator runtime mode.")
                return run_output_allocator()
            else:
                if self.use_output_allocator_outputs:  # users call OA context manager
                    if self.cudagraphs_enabled:
                        raise RuntimeError(
                            "Both CUDA Graphs and dynamic output allocation are enabled, which are incompatible runtime modes. Please disable one of the two."
                        )
                    logger.debug("Using the dynamic allocator runtime mode.")
                    return run_output_allocator()
                else:
                    logger.debug(
                        f"Using the standard execution runtime mode with cudagraphs={self.cudagraphs_enabled}."
                    )
                    return run_standard_execution()

    def enable_profiling(self, profiler: "trt.IProfiler" = None) -> None:
        """
        Enable TensorRT profiling. After calling this function, TensorRT will report
        time spent on each layer in stdout for each forward run.
        """
        self._check_initialized()

        if not self.context.profiler:
            self.context.profiler = trt.Profiler() if profiler is None else profiler

        self.profiling_enabled = True

    def disable_profiling(self) -> None:
        """
        Disable TensorRT profiling.
        """
        self._check_initialized()
        torch.cuda.synchronize()
        del self.context
        self.context = self.engine.create_execution_context()
        self.profiling_enabled = False

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        inspector = self.engine.create_engine_inspector()
        engine_json: str = inspector.get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        return engine_json

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """
        Validates the input shapes of the forward function has changed
        """
        # Representation of input shapes to a given model
        # Shapes are concatenated as so:
        # x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
        if not all(isinstance(t, torch.Tensor) for t in inputs):
            return True

        new_shape_key = "".join(
            str(tuple(t.shape)).replace(" ", "")
            for t in inputs
            if isinstance(t, torch.Tensor)
        )

        # If the new shape key differs from the existing one,
        # invalidate the old shape key and remove the CUDAGraph
        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key
            return True

        return False

    def are_output_tensors_unowned(self) -> bool:
        return self.output_tensors_are_unowned
