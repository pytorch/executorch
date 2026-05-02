from __future__ import annotations

import base64
import copy
import logging
import pickle
from typing import Any, List, Optional, Tuple, Union

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform
from torch_tensorrt._features import (
    ENABLED_FEATURES,
    for_all_methods,
    needs_torch_tensorrt_runtime,
)
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

SerializedTensorRTEngineFmt = List[
    Union[str, bytes]
]  # Aligned with  //core/runtime/register_jit_hooks.cpp
SerializedTorchTensorRTModuleFmt = Tuple[
    str, Optional[SerializedTensorRTEngineFmt], List[str], List[str]
]

ABI_TARGET_IDX = 0
NAME_IDX = 1
DEVICE_IDX = 2
ENGINE_IDX = 3
INPUT_BINDING_NAMES_IDX = 4
OUTPUT_BINDING_NAMES_IDX = 5
HW_COMPATIBLE_IDX = 6
SERIALIZED_METADATA_IDX = 7
TARGET_PLATFORM_IDX = 8
REQUIRES_OUTPUT_ALLOCATOR_IDX = 9
SERIALIZATION_LEN = 10


@for_all_methods(needs_torch_tensorrt_runtime)
class TorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """TorchTensorRTModule is a PyTorch module which encompasses an arbitrary TensorRT Engine.

    This module is backed by the Torch-TensorRT runtime and is fully compatible with both
    FX / Python deployments (just ``import torch_tensorrt`` as part of the application) as
    well as TorchScript / C++ deployments since TorchTensorRTModule can be passed to ``torch.jit.trace``
    and then saved.

    The forward function is simpily forward(*args: torch.Tensor) -> Tuple[torch.Tensor] where
    the internal implementation is ``return Tuple(torch.ops.tensorrt.execute_engine(list(inputs), self.engine))``

    > Note: TorchTensorRTModule only supports engines built with explicit batch

    Attributes:
        name (str): Name of module (for easier debugging)
        engine (torch.classes.tensorrt.Engine): Torch-TensorRT TensorRT Engine instance, manages [de]serialization, device configuration, profiling
        input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
        output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned
    """

    def __init__(
        self,
        serialized_engine: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),  # Assumes engine was built with default compilation settings if object not passed
        weight_name_map: Optional[dict[Any, Any]] = None,
        requires_output_allocator: bool = False,
    ):
        """Takes a name, target device, serialized TensorRT engine, and binding names / order and constructs
        a PyTorch ``torch.nn.Module`` around it. Uses the Torch-TensorRT runtime extension to run the engines

        If binding names are not provided, it is assumed that the engine binding names follow the following convention:

            - [symbol].[index in input / output array]
                - ex. [x.0, x.1, x.2] -> [y.0]

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

                with io.BytesIO() as engine_bytes:
                    engine_bytes.write(trt_engine.serialize())
                    engine_str = engine_bytes.getvalue()

                trt_module = TorchTensorRTModule(
                    engine_str,
                    input_binding_names=["x"],
                    output_binding_names=["output"],
                    name="my_module",
                    settings=CompilationSettings(device=torch.cuda.current_device)
                )

        """
        super(TorchTensorRTModule, self).__init__()

        if not isinstance(serialized_engine, bytearray):
            ValueError("Expected serialized engine as bytearray")

        self.input_binding_names = (
            input_binding_names if input_binding_names is not None else []
        )
        self.output_binding_names = (
            output_binding_names if output_binding_names is not None else []
        )
        self.name = name
        self.hardware_compatible = settings.hardware_compatible
        self.settings = copy.deepcopy(settings)
        self.weight_name_map = weight_name_map
        self.serialized_engine = serialized_engine
        self.engine = None
        self.requires_output_allocator = requires_output_allocator

        if (
            serialized_engine
            and not self.settings.lazy_engine_init
            and not self.settings.enable_cross_compile_for_windows
        ):
            self.setup_engine()

    def _pack_engine_info(self) -> List[str | bytes]:
        target_device = (
            self.settings.device
            if self.settings.device is not None
            else Device._current_device()
        )
        metadata = {
            "settings": self.settings,
            "weight_name_map": self.weight_name_map,
            "output_tensors_are_unowned": (
                False
                if self.engine is None
                else self.engine.are_output_tensors_unowned()
            ),
        }
        target_platform = (
            Platform.current_platform()
            if not self.settings.enable_cross_compile_for_windows
            else Platform.WIN_X86_64
        )  # Change to match target for engine

        engine_info: List[str | bytes] = [""] * SERIALIZATION_LEN
        engine_info[ABI_TARGET_IDX] = torch.ops.tensorrt.ABI_VERSION()
        engine_info[NAME_IDX] = (
            self.name + "_engine" if self.name != "" else "tensorrt_engine"
        )
        engine_info[DEVICE_IDX] = target_device._to_serialized_rt_device()
        assert self.serialized_engine
        engine_info[ENGINE_IDX] = self.serialized_engine

        engine_info[INPUT_BINDING_NAMES_IDX] = TorchTensorRTModule._pack_binding_names(
            self.input_binding_names
        )
        engine_info[OUTPUT_BINDING_NAMES_IDX] = TorchTensorRTModule._pack_binding_names(
            self.output_binding_names
        )
        engine_info[HW_COMPATIBLE_IDX] = str(int(self.hardware_compatible))
        engine_info[SERIALIZED_METADATA_IDX] = self.encode_metadata(metadata)
        engine_info[TARGET_PLATFORM_IDX] = target_platform._to_serialized_rt_platform()
        engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = str(
            int(self.requires_output_allocator)
        )

        return engine_info

    def get_streamable_device_memory_budget(self) -> Any:
        return self.engine.streamable_device_memory_budget

    def get_automatic_device_memory_budget(self) -> Any:
        return self.engine.automatic_device_memory_budget

    def get_device_memory_budget(self) -> Any:
        return self.engine.device_memory_budget

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        # Disable weight streaming for invalid budget size
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        self.engine.device_memory_budget = budget_bytes
        if self.engine.device_memory_budget != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = self.engine.device_memory_budget
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")

        return budget_bytes

    def _reset_captured_graph(self) -> None:
        self.engine.reset_captured_graph()

    def setup_engine(self) -> None:
        """
        Setup engine for a module which has deferred engine setup.

        Will setup the TensorRT engine for this module in the case that setup has been
        deferred. In the case that the engine has already been setup, will return without
        changing anything. Assumes that serialized engine and settings have already been passed
        to the module.
        """
        if self.engine is not None:
            return
        self.engine = torch.classes.tensorrt.Engine(self._pack_engine_info())

    def encode_metadata(self, metadata: Any) -> str:
        metadata = copy.deepcopy(metadata)
        dumped_metadata = pickle.dumps(metadata)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    @staticmethod
    def decode_metadata(encoded_metadata: bytes) -> Any:
        dumped_metadata = base64.b64decode(encoded_metadata.encode("utf-8"))
        metadata = pickle.loads(dumped_metadata)
        return metadata

    def get_extra_state(self) -> SerializedTorchTensorRTModuleFmt:
        if self.engine:
            return (
                self.name,
                self.engine.__getstate__(),
                self.input_binding_names,
                self.output_binding_names,
            )
        elif self.serialized_engine:
            engine_info = self._pack_engine_info()
            assert isinstance(engine_info[3], bytes)
            engine_info[ENGINE_IDX] = base64.b64encode(engine_info[3])
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
            )
        else:
            return (
                self.name,
                None,
                self.input_binding_names,
                self.output_binding_names,
            )

    def set_extra_state(self, state: SerializedTorchTensorRTModuleFmt) -> None:
        self.name = state[0]

        if state[1] is not None:
            serialized_engine_info: SerializedTensorRTEngineFmt = state[1]
            serialized_engine_info[ENGINE_IDX] = base64.b64decode(
                serialized_engine_info[ENGINE_IDX]
            )
            self.engine = torch.classes.tensorrt.Engine(serialized_engine_info)
            self.hardware_compatible = bool(
                int(serialized_engine_info[HW_COMPATIBLE_IDX])
            )
            self.requires_output_allocator = bool(
                int(serialized_engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])
            )

            serialized_metadata = serialized_engine_info[SERIALIZED_METADATA_IDX]
            assert isinstance(serialized_metadata, bytes)
            metadata = TorchTensorRTModule.decode_metadata(serialized_metadata)
            self.settings = metadata["settings"]
            self.weight_name_map = metadata["weight_name_map"]
            self.output_tensors_are_unowned = metadata["output_tensors_are_unowned"]
            self.engine.set_output_tensors_as_unowned(self.output_tensors_are_unowned)

        else:
            self.engine = None
            self.settings = CompilationSettings()
            self.hardware_compatible = False

        self.input_binding_names = state[2]
        self.output_binding_names = state[3]

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self.engine.use_pre_allocated_outputs = enable

    def set_use_output_allocator(self, enable: bool) -> None:
        self.engine.use_output_allocator_outputs = enable

    def forward(self, *inputs: Any) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Implementation of the forward pass for a TensorRT engine

        Args:
            *inputs (Union[torch.Tensor, int]): Inputs to the forward function

        Returns:
            torch.Tensor or Tuple(torch.Tensor): Result of the engine computation
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been setup yet.")

        assert len(inputs) == len(
            self.input_binding_names
        ), f"Wrong number of inputs, expected {len(self.input_binding_names)} got {len(inputs)}."

        # If the inputs are not Torch Tensors, which can occur in scenarios such as shape tensors
        # which are outputs of a preceding Torch subgraph (where the Dynamic input may be an integer)
        # directly cast the input to a Torch Tensor.
        #
        # This also avoids the need for type-checking inputs, since they are now explicitly casted to Torch tensors
        input_tensors: List[torch.Tensor] = [
            (i if isinstance(i, torch.Tensor) else torch.tensor(i).cuda())
            for i in inputs
        ]

        outputs: List[torch.Tensor] = torch.ops.tensorrt.execute_engine(
            list(input_tensors), self.engine
        )

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def enable_profiling(
        self,
        profiling_results_dir: Optional[str] = None,
        profile_format: str = "perfetto",
    ) -> None:
        """Enable the profiler to collect latency information about the execution of the engine

        Traces can be visualized using https://ui.perfetto.dev/ or compatible alternatives

        Keyword Arguments:
            profiling_results_dir (str): Absolute path to the directory to sort results of profiling.
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        if profiling_results_dir is not None:
            self.engine.profile_path_prefix = profiling_results_dir
        assert profile_format in ["trex", "perfetto"]
        self.engine.enable_profiling()
        self.engine.set_profile_format(profile_format)

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self.engine.set_output_tensors_as_unowned(enabled)

    def are_output_tensors_unowned(self) -> bool:
        return self.engine.are_output_tensors_unowned()  # type: ignore[no-any-return]

    def disable_profiling(self) -> None:
        """Disable the profiler"""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        self.engine.disable_profiling()

    def get_layer_info(self) -> str:
        """Get a JSON string containing the layer information encoded by the TensorRT engine in this module

        Returns:

            str: A JSON string which contains the layer information of the engine incapsulated in this module
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        layer_info: str = self.engine.get_engine_layer_info()
        return layer_info

    def dump_layer_info(self) -> None:
        """Dump layer information encoded by the TensorRT engine in this module to STDOUT"""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        self.engine.dump_engine_layer_info()

    @staticmethod
    def _pack_binding_names(binding_names: List[str]) -> str:
        delim = torch.ops.tensorrt.SERIALIZED_ENGINE_BINDING_DELIM()[0]
        packed_bindings: str = delim.join(binding_names)
        return packed_bindings
