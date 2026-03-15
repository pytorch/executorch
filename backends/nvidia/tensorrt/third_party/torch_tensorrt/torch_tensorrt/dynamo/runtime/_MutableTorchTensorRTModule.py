import inspect
import logging
import warnings
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Dict, Iterator, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch_tensorrt
from torch.export._trace import _export
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
from torch_tensorrt.dynamo._refit import refit_module_weights
from torch_tensorrt.dynamo.utils import (
    check_output_equal,
    deallocate_module,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


class RefitFlag(Enum):
    UNKNOWN = auto()
    NEEDS_REFIT = auto()
    NEEDS_RECOMPILE = auto()
    LIVE = auto()


class RefitState:
    _state: RefitFlag = RefitFlag.NEEDS_RECOMPILE

    def set_state(self, state: RefitFlag) -> None:
        if isinstance(state, RefitFlag):
            self._state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def get_state(self) -> RefitFlag:
        return self._state


class DynamicShapeOutOfRangeException(Exception):
    pass


class MutableTorchTensorRTModule(object):
    """
    Initialize a MutableTorchTensorRTModule to seamlessly manipulate it like a regular PyTorch module.
    All TensorRT compilation and refitting processes are handled automatically as you work with the module.
    Any changes to its attributes or loading a different state_dict will trigger refitting or recompilation,
    which will be managed during the next forward pass.

    The MutableTorchTensorRTModule takes a PyTorch module and a set of configuration settings for the compiler.
    Once compilation is complete, the module maintains the connection between the TensorRT graph module and the original PyTorch module.
    Any modifications made to the MutableTorchTensorRTModule will be reflected in both the TensorRT graph module and the original PyTorch module.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        *,
        device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
        use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
        immutable_weights: bool = False,
        strict: bool = True,
        prefer_deferred_runtime_asserts_over_guards: bool = False,
        weight_streaming_budget: Optional[int] = None,
        enabled_precisions: Union[
            Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]
        ] = _defaults.ENABLED_PRECISIONS,
        **kwargs: Any,
    ) -> None:
        """

        Arguments:
            pytorch_model (torch.nn.module): Source module that needs to be accelerated

        Keyword Arguments:
            device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

                device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

            disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
            assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
            sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
            enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
            immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable.
            capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
            num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
            workspace_size (int): Maximum size of workspace given to TensorRT
            dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
            dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
            dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
            truncate_double (bool): Truncate weights provided in double (float64) to float32
            require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
            min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
            torch_executed_ops (Collection[Target]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
            torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
            pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
            max_aux_stream (Optional[int]): Maximum streams in the engine
            version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
            optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
            use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
            use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optimal. Use the global paritioner (``False``) if looking for best performance
            enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
            dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
            hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
            timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
            lazy_engine_init (bool): Defer setting up engines until the compilation of all engines is complete. Can allow larger models with multiple graph breaks to compile but can lead to oversubscription of GPU memory at runtime.
            enabled_precisions (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
            **kwargs: Any,
        Returns:
            MutableTorchTensorRTModule
        """
        # The order to initialize this module is
        # 1. Set init_finished to False
        # 2. Initialize all attributes
        # 3. Add the module base class
        # 4. Set the init_finished to True
        # After initialization, no new attribute should be added to the module __dict__
        # Otherwise, it will cause undefined behavior

        object.__setattr__(self, "init_finished", False)
        self.refit_state = RefitState()
        self.pytorch_model = _make_refit_change_trigger(pytorch_model, self.refit_state)
        self.original_model = pytorch_model
        if pytorch_model.training:
            logger.warning(
                "The model may be in training mode, which may affect the performance of the compiled model!"
            )
        # Process settings
        self.gm: Any = None
        self.exp_program: Any = None
        self.arg_inputs: tuple[Any, ...] = tuple()
        self.kwarg_inputs: dict[str, Any] = {}
        self.additional_settings = kwargs
        self.strict = strict
        self.prefer_deferred_runtime_asserts_over_guards = (
            prefer_deferred_runtime_asserts_over_guards
        )
        self.use_python_runtime = use_python_runtime
        self.trt_device = to_torch_tensorrt_device(device)
        assert (
            not immutable_weights
        ), "`immutable_weights has to be False for a MutableTorchTensorRTModule"

        self.arg_dynamic_shapes: Optional[tuple[Any]] = None
        self.kwarg_dynamic_shapes: Optional[dict[Any, Any]] = None
        self.serializable_dynamic_shapes_dims: dict[str, tuple[str, int, int]] = {}
        self.run_info: Optional[tuple[Any, ...]] = None
        self.state_dict_metadata: dict[str, torch.Size] = {}
        self._store_state_dict_metadata()
        self.enable_weight_streaming = (
            kwargs["enable_weight_streaming"]
            if "enable_weight_streaming" in kwargs
            else False
        )
        self.weight_streaming_ctx = None
        self.weight_streaming_budget = weight_streaming_budget
        if self.enable_weight_streaming:
            if weight_streaming_budget is None:
                logger.warning(
                    "Weight stremaing budget is not set. Using auto weight streaming budget"
                )
        self.enabled_precisions = enabled_precisions

        cls = self.__class__
        self.__class__ = type(
            self.original_model.__class__.__name__,
            (cls, pytorch_model.__class__),
            {},
        )
        self.init_finished = True

    def set_expected_dynamic_shape_range(
        self,
        args_dynamic_shape: tuple[dict[Any, Any]],
        kwargs_dynamic_shape: dict[str, Any],
    ) -> None:
        """
        Set the dynamic shape range. The shape hint should EXACTLY follow arg_inputs and kwarg_inputs passed to the forward function
        and should not omit any entries (except None in the kwarg_inputs). If there is a nested dict/list in the input, the dynamic shape for that entry should also be an nested dict/list.
        If the dynamic shape is not required for an input, an empty dictionary should be given as the shape hint for that input.
        Note that you should exclude keyword arguments with value None as those will be filtered out.

        Example:
        def forward(a, b, c=0, d=0):
            pass

        seq_len = torch.export.Dim("seq_len", min=1, max=10)
        args_dynamic_shape = ({0: seq_len}, {}) # b does not have a dynamic shape
        kwargs_dynamic_shape = {'c': {0, seq_len}, 'd': {}} # d does not have a dynamic shape
        set_expected_dynamic_shape_range(args_dynamic_shape, kwargs_dynamic_shape)
        # Later when you call the function
        forward(*(a, b), **{c:..., d:...})

        Reference: https://pytorch.org/docs/stable/export.html#expressing-dynamism
        Arguments:
            args_dynamic_shape (tuple[dict[Any, Any]]): Dynamic shape hint for the arg_inputs,
            kwargs_dynamic_shape: (dict[str, Any]): Dynamic shape hint for the kwarg_inputs
        """
        assert isinstance(
            args_dynamic_shape, tuple
        ), f"args dynamic shape has to be a tuple, but got {type(args_dynamic_shape)}"
        assert isinstance(
            kwargs_dynamic_shape, dict
        ), f"args dynamic shape has to be a dictionary, but got {type(kwargs_dynamic_shape)}"
        self.kwarg_dynamic_shapes = kwargs_dynamic_shape
        self.arg_dynamic_shapes = args_dynamic_shape

        # Clear cached inputs
        self.arg_inputs = tuple()
        self.kwarg_inputs = {}

        self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)

    def _get_total_dynamic_shapes(self) -> Union[dict[str, Any], None]:
        if not self.arg_dynamic_shapes and not self.kwarg_dynamic_shapes:
            return None
        total_dynamic_shape = {}
        if self.arg_dynamic_shapes:
            signature = list(
                inspect.signature(self.original_model.forward).parameters.keys()
            )
            for i, arg in enumerate(self.arg_dynamic_shapes):
                total_dynamic_shape[signature[i]] = arg

        if self.kwarg_dynamic_shapes:
            for kwargs, kwargs_dynamic_shape in self.kwarg_dynamic_shapes.items():
                total_dynamic_shape[kwargs] = kwargs_dynamic_shape

        return total_dynamic_shape

    def _store_state_dict_metadata(self) -> None:
        for k, v in self.original_model.state_dict().items():
            self.state_dict_metadata[k] = v.shape

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        self.refit_state.set_state(RefitFlag.NEEDS_REFIT)
        self.original_model.load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _transform_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.nn.Parameter]:
        return {k: torch.nn.Parameter(v, requires_grad=False) for k, v in sd.items()}

    def update_refit_condition(self) -> None:
        # 2-stage check to determine whether the module should be intact, refitted, or recompiled.

        # Default refit
        self.refit_state.set_state(RefitFlag.NEEDS_REFIT)

        # Run the same inputs through pytorch model and compare the result to previous run of graph module
        # to determine whether refit/recompilation is needed. If the output is the same, no further process needed.
        if self.run_info:
            args, kwargs, result = self.run_info
            self.original_model.to(to_torch_device(self.trt_device))
            new_result = self.original_model(*args, **kwargs)
            deallocate_module(self.original_model, delete_module=False)
            if check_output_equal(result, new_result):
                self.refit_state.set_state(RefitFlag.LIVE)
                return

        # Since we do not have access to the previous state_dict, we can only use state_dict_metadata
        # to determine whether the keys or weight shape is changed.
        sd, sd_meta = self.original_model.state_dict(), self.state_dict_metadata
        if sd.keys() != sd_meta.keys():
            # If keys are not identical, recompile.
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            return

        for k in sd.keys():
            if sd[k].shape != sd_meta[k]:
                # If weight shapes are not identical, recompile.
                self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
                return

        return

    def refit_gm(self) -> None:
        """
        Refit the TRT graph module with any updates.
        This function should be called whenever the weight values get changed but the weight structure remains
        the same.
        MutableTorchTensorRTModule automatically catches weight value updates and call this function to refit the module.
        If it fails to catch the changes, please call this function manually to update the TRT graph module.
        """

        if self.exp_program is None:
            self.original_model.to(to_torch_device(self.trt_device))
            self.exp_program = self.get_exported_program()
        else:
            self.exp_program._state_dict = (
                MutableTorchTensorRTModule._transform_state_dict(
                    self.original_model.state_dict()
                )
            )
            self.exp_program.module().to(to_torch_device(self.trt_device))
        self.gm = refit_module_weights(
            self.gm,
            self.exp_program,
            self.arg_inputs,
            self.kwarg_inputs,
            use_weight_map_cache=True,
            in_place=True,
        )

        deallocate_module(self.original_model, delete_module=False)

    def get_exported_program(self) -> torch.export.ExportedProgram:

        def export_fn() -> torch.export.ExportedProgram:
            if self.prefer_deferred_runtime_asserts_over_guards:
                return _export(
                    self.original_model,
                    self.arg_inputs,
                    kwargs=self.kwarg_inputs,
                    dynamic_shapes=self._get_total_dynamic_shapes(),
                    strict=self.strict,
                    prefer_deferred_runtime_asserts_over_guards=self.prefer_deferred_runtime_asserts_over_guards,
                )
            else:
                return torch.export.export(
                    self.original_model,
                    self.arg_inputs,
                    kwargs=self.kwarg_inputs,
                    dynamic_shapes=self._get_total_dynamic_shapes(),
                    strict=self.strict,
                )

        # Check if any quantization precision is enabled
        if self.enabled_precisions and any(
            precision in self.enabled_precisions
            for precision in (torch.float8_e4m3fn, torch.int8, torch.float4_e2m1fn_x2)
        ):
            try:
                from modelopt.torch.quantization.utils import export_torch_mode

                assert torch.ops.tensorrt.quantize_op.default
            except Exception as e:
                logger.warning(
                    "Unable to import quantization op. Please install modelopt library (https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#installation) to add support for compiling quantized models"
                )
            with export_torch_mode():
                return export_fn()
        else:
            return export_fn()

    def compile(self) -> None:
        """
        (Re)compile the TRT graph module using the PyTorch module.
        This function should be called whenever the weight structure get changed (shape, more layers...)
        MutableTorchTensorRTModule automatically catches weight value updates and call this function to recompile.
        If it fails to catch the changes, please call this function manually to recompile the TRT graph module.
        """
        # Export the module
        self.original_model.to(to_torch_device(self.trt_device))
        self.exp_program = self.get_exported_program()
        self.gm = dynamo_compile(
            self.exp_program,
            arg_inputs=self.arg_inputs,
            kwarg_inputs=self.kwarg_inputs,
            immutable_weights=False,
            use_python_runtime=self.use_python_runtime,
            enabled_precisions=self.enabled_precisions,
            **self.additional_settings,
        )
        if self.additional_settings.get("offload_module_to_cpu", False):
            deallocate_module(self.original_model, delete_module=False)
        if self.enable_weight_streaming:
            self.set_weight_streaming_ctx(self.weight_streaming_budget)

    def set_weight_streaming_ctx(self, requested_budget: Optional[int] = None) -> None:
        """
        Set the weight streaming budget. If budget is not set, then automatic weight streaming budget
        is used.
        """
        self.weight_streaming_ctx = torch_tensorrt.runtime.weight_streaming(self.gm)
        requested_budget = (
            requested_budget
            if requested_budget is not None
            else self.weight_streaming_ctx.get_automatic_weight_streaming_budget()
        )
        self.weight_streaming_ctx.device_budget = requested_budget

    def _validate_inputs(self, *args: Any, **kwargs: Any) -> None:

        if not self.arg_inputs and not self.kwarg_inputs:
            logger.info("First time compilation initiated. This may take some time.")
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            self._store_inputs(args, kwargs)
            if self.arg_dynamic_shapes or self.kwarg_dynamic_shapes:
                if not self._validates_dynamic_hints():
                    logger.warning(
                        "Invalid dynamic shape hint. Compiling module for the provided input shapes (static)"
                    )
                    self.arg_dynamic_shapes = None
                    self.kwarg_dynamic_shapes = None
            return

        # If input does not equal or does not fall into dynamic shape range, recompile the engine
        try:
            if not MutableTorchTensorRTModule._check_inputs_shape(
                self.arg_inputs, args, dynamic_shapes=self.arg_dynamic_shapes
            ) or not MutableTorchTensorRTModule._check_inputs_shape(
                self.kwarg_inputs, kwargs, dynamic_shapes=self.kwarg_dynamic_shapes
            ):
                logger.info("Input change detected.")
                self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
                self._store_inputs(args, kwargs)
        except DynamicShapeOutOfRangeException as e:
            logger.info("Input change detected.")
            logger.warning(e)
            logger.warning(
                "Provided inputs are outside the set expected shape range, recompiling module for the provided input shapes (static)"
            )
            self.arg_dynamic_shapes = None
            self.kwarg_dynamic_shapes = None
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            self._store_inputs(args, kwargs)

    def _validates_dynamic_hints(self) -> bool:
        if self.arg_dynamic_shapes is None:
            if self.arg_inputs:
                logger.warning("arg_dynamic_shape is not provided!")
        else:
            if len(self.arg_dynamic_shapes) != len(self.arg_inputs):
                logger.warning(
                    f"Warning: The length of arg_inputs is {len(self.arg_inputs)} but the length of arg_dynamic_shape is {len(self.arg_dynamic_shapes)}!"
                )
                return False

        if self.kwarg_dynamic_shapes is None:
            if self.kwarg_inputs:
                logger.warning("kwarg_dynamic_shape is not provided!")
        else:
            if self.kwarg_dynamic_shapes.keys() != self.kwarg_inputs.keys():
                logger.warning(
                    f"kwarg_inputs has {list(self.kwarg_inputs.keys())} but kwarg_dynamic_shape has {list(self.kwarg_dynamic_shapes.keys())}! You may need to exclude keyword arguments with value None."
                )
                return False

        return True

    def _store_inputs(self, arg_inputs: Any, kwarg_inputs: Any) -> None:
        self.arg_inputs = arg_inputs
        self.kwarg_inputs = kwarg_inputs

    @staticmethod
    def _process_kwarg_inputs(inputs: Any) -> Any:
        # Process kwarg inputs to be acceptable for Torch-TensorRT
        if isinstance(inputs, dict):
            # None should be excluded. AOT compile also does not allow dynamic control flow, bool is also excluded.
            return {
                k: MutableTorchTensorRTModule._process_kwarg_inputs(v)
                for k, v in inputs.items()
                if (v is not None)
            }
        elif isinstance(inputs, (torch.Tensor, bool)):
            return inputs
        elif isinstance(inputs, (int, float, np.ndarray)):
            return torch.tensor(inputs)
        elif isinstance(inputs, (list, tuple)):
            if None not in inputs:
                return type(inputs)(
                    [
                        MutableTorchTensorRTModule._process_kwarg_inputs(v)
                        for v in inputs
                    ]
                )

        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the input. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "Direct calls to {self.__class__}.forward() are currently broken by due to https://github.com/pytorch/pytorch/issues/157183. Either call {self.__class__}(...) directly or use {self.__class__}._forward as a work around"
        )
        return self._forward(*args, **kwargs)

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        # Step 1: Check whether the input shape has changed
        kwargs = MutableTorchTensorRTModule._process_kwarg_inputs(kwargs)
        self._validate_inputs(*args, **kwargs)

        # Step 2: If the flag is unknown, it could be a recompile or refit.
        if self.refit_state.get_state() == RefitFlag.UNKNOWN:
            # Update the flag
            self.update_refit_condition()

        # Step 3: Refit/recompile accordingly
        if self.refit_state.get_state() == RefitFlag.NEEDS_RECOMPILE:
            logger.info("(Re)Compiling the engine...")
            self.compile()
            self._store_state_dict_metadata()
            self.refit_state.set_state(RefitFlag.LIVE)

        elif self.refit_state.get_state() == RefitFlag.NEEDS_REFIT:
            logger.info("Model weight change detected. Refitting the module...")
            try:
                self.refit_gm()
            except Exception as e:
                logger.error(e)
                logger.error("Model refit failed. Recompiling the graph module.")
                self.compile()
                self._store_state_dict_metadata()
            self.refit_state.set_state(RefitFlag.LIVE)

        weight_streaming_ctx = (
            self.weight_streaming_ctx if self.enable_weight_streaming else None
        )
        result = self.gm(*args, **kwargs)
        # Storing inputs and outputs for verification when the state is unknown
        self.run_info = (args, kwargs, result)
        return result

    def to(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "Trying to move the original PyTorch model. This will cause CPU offloading failing and increase GPU memory usage."
            + "If this is absolute necessary, please call module.pytorch_model.to(...) \n"
            + "The model is still on the original device."
        )

    @property
    def device(self) -> torch.device:
        return to_torch_device(self.trt_device)

    def __deepcopy__(self, memo: Any) -> Any:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "pytorch_model":
                setattr(result, k, deepcopy(v, memo))
        result.pytorch_model = _make_refit_change_trigger(
            result.original_model, result.refit_state
        )
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Due to https://github.com/pytorch/pytorch/issues/157183, we cannot use forward call, use _forward as a workaround.
        # This is a temporary fix.
        return self._forward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            # this object has it
            return getattr(self, name)

        if "pytorch_model" not in self.__dict__:
            raise AttributeError(
                "Module is not properly initiated. Pytorch model is not found in the module."
            )

        return getattr(self.pytorch_model, name)

    def __delattr__(self, name: str) -> Any:
        if name in self.__dict__:
            # this object has it
            super().__delattr__(name)

        return self.pytorch_model.__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # When the module finished initialization, any modification to attributes that does not exist
        # in __dict__ will be handled in pytorch module.
        if self.init_finished:
            if name in self.__dict__:
                object.__setattr__(self, name, value)
            else:
                # Capture attribute change
                self.refit_state.set_state(RefitFlag.UNKNOWN)
                # We want to make sure the original PyTorch model does not have a trigger wrapper
                value = recursively_remove_trigger(value)
                setattr(self.original_model, name, value)
        else:
            object.__setattr__(self, name, value)

    @staticmethod
    def _check_inputs_shape(
        input1: Any,
        input2: Any,
        dynamic_shapes: Any = None,
    ) -> bool:

        if isinstance(input1, (tuple, list)):
            if len(input1) != len(input2):
                return False
            for (i, a), b in zip(enumerate(input1), input2):
                if type(a) != type(b):
                    return False
                if isinstance(a, bool) and a != b:
                    return False
                elif isinstance(a, torch.Tensor) and a.shape != b.shape:
                    if dynamic_shapes is None:
                        logger.warning(
                            "Dynamic shape is not properly set but the input shape is changed!"
                        )
                        return False
                    else:
                        tensor_dynamic_shape = dynamic_shapes[i]
                        if not MutableTorchTensorRTModule._check_tensor_shapes_with_dynamic_shapes(
                            a, b, tensor_dynamic_shape
                        ):
                            return False

        elif isinstance(input1, dict):
            if input1.keys() != input2.keys():
                return False
            for ka, va in input1.items():
                vb = input2[ka]
                if type(va) != type(vb):
                    return False
                if isinstance(va, bool) and va != vb:
                    return False
                elif isinstance(va, torch.Tensor) and va.shape != vb.shape:
                    if dynamic_shapes is None:
                        logger.warning(
                            "Dynamic shape is not properly set but the input shape is changed!"
                        )
                        return False
                    else:
                        tensor_dynamic_shape = dynamic_shapes[ka]
                        if not MutableTorchTensorRTModule._check_tensor_shapes_with_dynamic_shapes(
                            va, vb, tensor_dynamic_shape
                        ):
                            return False
                elif isinstance(
                    va, (list, tuple, dict)
                ) and not MutableTorchTensorRTModule._check_inputs_shape(
                    va, vb, dynamic_shapes[ka] if dynamic_shapes else None
                ):
                    return False
        return True

    @staticmethod
    def _check_tensor_shapes_with_dynamic_shapes(
        input_1: torch.tensor, input_2: torch.tensor, dynamic_shape: dict[int, Any]
    ) -> bool:
        for (i, axis_0), axis_1 in zip(enumerate(input_1.shape), input_2.shape):
            if axis_0 != axis_1:
                if i not in dynamic_shape:
                    logger.warning(
                        "Dynamic shape does not include the axis on which input changes!"
                    )
                    return False
                dyn = dynamic_shape[i]
                if axis_1 > dyn.max or axis_1 < dyn.min:
                    raise DynamicShapeOutOfRangeException(
                        f"Dimension ({i}) of new input tensor is not the range of supported shapes (saw: ({axis_1}), expected: [{dyn.min}, {dyn.max}])"
                    )

        return True

    def serialize_dynamic_shapes(self) -> None:
        dims = self.serializable_dynamic_shapes_dims

        def resursivly_serialize_dynamic_shape(obj: Any) -> None:
            if isinstance(obj, dict):
                for axis, v in obj.items():
                    if isinstance(v, torch.export.dynamic_shapes._Dim):
                        name = str(v).split("'")[1].split(".")[-1]
                        # We use string of the hash to be the unique identifier of Dim object
                        dims.setdefault(str(hash(v)), (name, v.min, v.max))
                        obj[axis] = str(hash(v))
                    else:
                        resursivly_serialize_dynamic_shape(v)
            if isinstance(obj, (tuple, list)):
                for v in obj:
                    resursivly_serialize_dynamic_shape(v)

        resursivly_serialize_dynamic_shape(self.arg_dynamic_shapes)
        resursivly_serialize_dynamic_shape(self.kwarg_dynamic_shapes)

    def deserialize_dynamic_shapes(self) -> None:
        dims = self.serializable_dynamic_shapes_dims

        def resursivly_deserialize_dynamic_shape(obj: Any) -> None:
            if isinstance(obj, dict):
                for axis, v in obj.items():
                    if isinstance(v, str):
                        obj[axis] = torch.export.Dim(
                            dims[v][0], min=dims[v][1], max=dims[v][2]
                        )
                    else:
                        resursivly_deserialize_dynamic_shape(v)
            if isinstance(obj, (tuple, list)):
                for v in obj:
                    resursivly_deserialize_dynamic_shape(v)

        resursivly_deserialize_dynamic_shape(self.arg_dynamic_shapes)
        resursivly_deserialize_dynamic_shape(self.kwarg_dynamic_shapes)

    @staticmethod
    def save(module: Any, path: str) -> None:
        # Cast the object back to MutableTorchTensorRTModule to save
        assert (
            not module.use_python_runtime
        ), "Python runtime does not support serialization. Save failed."
        module.init_finished = False
        module.__class__ = MutableTorchTensorRTModule
        exp_program = module.exp_program
        module.pytorch_model = None
        module.exp_program = None
        module.serialize_dynamic_shapes()
        torch.save(module, path, pickle_protocol=4)
        # Restore deleted attributes
        module.exp_program = exp_program
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )
        cls = module.__class__
        module.__class__ = type(
            module.original_model.__class__.__name__,
            (cls, module.original_model.__class__),
            {},
        )

        module.init_finished = True

    @staticmethod
    def load(path: str) -> Any:
        # When the model get saved, init_finished is set to False.
        # Class is restored to MutableTorchTensorRTModule, and some attribute is deleted
        module = torch.load(path, weights_only=False)
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )
        module.original_model.to(to_torch_device(module.device))
        module.exp_program = torch.export.export(
            module.original_model, module.arg_inputs, kwargs=module.kwarg_inputs
        )
        deallocate_module(module.original_model, delete_module=False)
        cls = module.__class__
        module.__class__ = type(
            module.original_model.__class__.__name__,
            (cls, module.original_model.__class__),
            {},
        )
        module.deserialize_dynamic_shapes()
        module.init_finished = True
        return module

    def _reset_stateful_cache(obj: Any) -> None:
        """
        Does nothing. Support Huggingface CPU offload hooks. Override the huggingface cache reset function because we don't want the TRT module to be handled by HuggingFace.
        """
        return


def recursively_remove_trigger(obj: Any) -> Any:
    # Not safe: If the object has a circular reference (such as a doubly linkded list), this will cause infinite recursion
    if obj.__class__.__name__ == "ChangeTriggerWrapper":
        obj = obj.instance

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursively_remove_trigger(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = recursively_remove_trigger(v)
    else:
        if not hasattr(obj, "__dict__") or isinstance(obj, (type, torch.Tensor)):
            return obj
        for k, v in obj.__dict__.items():
            if k[:2] != "__" or k[-2:] != "__":
                # We don't want to touch some built in attribute such as __dict__
                setattr(obj, k, recursively_remove_trigger(v))

    return obj


def _make_refit_change_trigger(obj: object, refit_state: RefitState) -> Any:
    subclass: type = obj.__class__

    class ChangeTriggerWrapper(subclass):  # type: ignore
        # The reason why we want to inherent to the subclass is that we want the ChangeTriggerWrapper shares all functions
        # that an ordinary object has. In this way attributes accessed inside a function will be from the __getattr__function
        # of ChangeTriggerWrapper, instead of the object itself, thus be recursively wrapped by ChangeTriggerWrapper.

        def __init__(self, obj: Any):
            object.__setattr__(self, "instance", obj)

        def __getattr__(
            self, name: str
        ) -> Any:  # Called when the attribute does not exist
            obj = getattr(self.instance, name)
            if isinstance(obj, torch.nn.Parameter):
                # Whenever the user retrieve an attribute that could be related to weights, we set the state to UNKNOWN
                self._on_change()
            if (
                hasattr(obj, "__dict__") or isinstance(obj, (torch.nn.ModuleList, list))
            ) and not isinstance(
                obj, ChangeTriggerWrapper
            ):  # prevent nesting wrapper
                return _make_refit_change_trigger(obj, refit_state)
            return obj

        def __setattr__(self, name: str, value: Any) -> None:
            # If we need to set __dict__ or instance, we directly set it to the trigger wrapper.
            # Enable setting __dict__ is because PyTorch proxy uses __new__ to initialize a shallow copy
            # of a module and explicit set the __dict__. If we don't set __dict__ it will get infinite recursion.
            if name in ["__dict__", "instance"]:
                object.__setattr__(self, name, value)
                return
            self._on_change()
            # We want to make sure the original PyTorch model does not have a trigger wrapper
            value = recursively_remove_trigger(value)
            setattr(self.instance, name, value)

        def __delattr__(self, name: str) -> None:
            self._on_change()
            delattr(
                self.instance,
                name,
            )

        def _on_change(self) -> None:
            refit_state.set_state(RefitFlag.UNKNOWN)
            logger.info(
                "Attribute modification detected. The module will be refitted later."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance(*args, **kwargs)

        def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance._call_impl(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance.forward(*args, **kwargs)

        def __setitem__(self, item: str, value: Any) -> None:
            self._on_change()
            # We want to make sure the original PyTorch model does not have a trigger wrapper
            value = recursively_remove_trigger(value)
            self.instance.__setitem__(item, value)

        def __getitem__(self, items: str) -> Any:
            obj = self.instance.__getitem__(items)
            if isinstance(obj, ChangeTriggerWrapper):
                return obj
            return _make_refit_change_trigger(obj, refit_state)

        def __len__(self) -> int:
            return len(self.instance)

        def __iter__(self) -> Iterator[Any]:
            return iter(self.instance)

    return ChangeTriggerWrapper(obj)
