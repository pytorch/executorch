import base64
from collections import defaultdict
from typing import Any, List

import torch
from torch_tensorrt.dynamo.utils import input_is_dynamic, unwrap_tensor_shape


@torch.library.register_fake("tensorrt::execute_engine")  # type: ignore
def fake_tensorrt_execute_engine(
    inputs: List[torch.Tensor], fake_trt_engine: Any
) -> Any:
    """
    We infer outputs using the TRT engine and inputs and return fake tensors in this meta kernel.
    """
    # Here's what we are doing
    # 1) Check if inputs are dynamic (they have sym ints in their shapes)
    # 2) For dynamic inputs, we gather min_input_shape and max_input shape for all inputs
    # 3) For the above min and max input shape, capture the corresponding min and max output shape using TensorRT's set/get shapes mechanism
    # 4) Create a new symbolic fake tensor using min and max output shape for each output and return them
    # 5) For static inputs, the output shape will be static and we won't need to create sym ints
    is_dynamic_execution = input_is_dynamic(inputs)
    if is_dynamic_execution:
        modes = ["min", "max", "opt"]
    else:
        modes = ["opt"]

    # Get the TRTEngine class and infer output shapes based on input shapes
    trt_engine = fake_trt_engine.real_obj
    outputs_mode_dict = defaultdict(list)
    for mode in modes:
        input_shapes = [unwrap_tensor_shape(input, mode=mode) for input in inputs]
        proxy_outputs = trt_engine.infer_outputs(input_shapes)
        outputs_mode_dict[mode].extend(proxy_outputs)

    # Store the number of outputs
    if {"min", "max"}.issubset(outputs_mode_dict):
        assert len(outputs_mode_dict["min"]) == len(outputs_mode_dict["max"])
        num_outputs = len(outputs_mode_dict["min"])
    elif "opt" in outputs_mode_dict:
        num_outputs = len(outputs_mode_dict["opt"])

    fake_outputs = []
    for out_idx in range(num_outputs):
        output_shape = []
        if is_dynamic_execution:
            # Create output symbolic shape using unbacked symint.
            # Note: We can't establish a relationship b/w incoming input symbolic shape (eg: s0)
            # and TensorRT's output shape (represented as unbacked u0). This situation doesn't seem
            # to affect compilation results / serialization during our testing.
            output_min_shape = outputs_mode_dict["min"][out_idx].size()
            output_opt_shape = outputs_mode_dict["opt"][out_idx].size()
            output_max_shape = outputs_mode_dict["max"][out_idx].size()

            ctx = torch._custom_ops.get_ctx()
            for min_val, opt_val, max_val in zip(
                output_min_shape, output_opt_shape, output_max_shape
            ):
                if min_val != max_val:
                    output_sym_int = ctx.new_dynamic_size(min=min_val, max=max_val)
                    # Update var to val (hint)
                    output_sym_int_shape_env = output_sym_int.node.shape_env
                    output_sym_int_shape_env.set_unbacked_var_to_val(
                        output_sym_int.node.expr, opt_val
                    )
                    output_shape.append(output_sym_int)
                else:
                    output_shape.append(min_val)
        else:
            output_shape.extend(outputs_mode_dict["opt"][out_idx].size())
        fake_outputs.append(
            torch.empty(
                output_shape,
                dtype=outputs_mode_dict["opt"][out_idx].dtype,
                device=inputs[0].device,
            )
        )
    return fake_outputs


@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self, engine_info: List[str]) -> None:
        self.version = engine_info[torch.ops.tensorrt.ABI_TARGET_IDX()]
        self.name = engine_info[torch.ops.tensorrt.NAME_IDX()]
        self.device_info = engine_info[torch.ops.tensorrt.DEVICE_IDX()]
        self.serialized_engine = engine_info[torch.ops.tensorrt.ENGINE_IDX()]
        self.in_binding_names = engine_info[
            torch.ops.tensorrt.INPUT_BINDING_NAMES_IDX()
        ]
        self.out_binding_names = engine_info[
            torch.ops.tensorrt.OUTPUT_BINDING_NAMES_IDX()
        ]
        self.hardware_compatible = engine_info[torch.ops.tensorrt.HW_COMPATIBLE_IDX()]
        self.serialized_metadata = engine_info[
            torch.ops.tensorrt.SERIALIZED_METADATA_IDX()
        ]
        self.requires_output_allocator = engine_info[
            torch.ops.tensorrt.REQUIRES_OUTPUT_ALLOCATOR_IDX()
        ]
        self.target_platform = engine_info[torch.ops.tensorrt.TARGET_PLATFORM_IDX()]

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        engine_idx = torch.ops.tensorrt.ENGINE_IDX()
        engine_info = [info[1] for info in flattened_tq]
        engine_info[engine_idx] = base64.b64decode(engine_info[engine_idx])

        return cls(engine_info)

    def enable_profiling(self) -> Any:
        pass

    def disable_profiling(self) -> Any:
        pass

    def dump_engine_layer_info_to_file(self, path: str) -> Any:
        pass

    def dump_engine_layer_info(self) -> Any:
        pass

    def get_engine_layer_info(self) -> Any:
        pass

    def profile_path_prefix_getter(self) -> Any:
        pass

    def profile_path_prefix_setter(self) -> Any:
        pass

    def device_memory_budget_getter(self) -> Any:
        pass

    def device_memory_budget_setter(self) -> Any:
        pass

    def streamable_device_memory_budget_getter(self) -> Any:
        pass

    def automatic_device_memory_budget_getter(self) -> Any:
        pass

    def infer_outputs(self, input_shapes: List[Any]) -> Any:
        pass

    def reset_captured_graph(self) -> Any:
        pass

    def __setstate__(self, serialized_state: List[str]) -> Any:
        pass

    def __getstate__(self) -> Any:
        pass


@torch.library.custom_op(  # type: ignore
    "tensorrt::no_op_placeholder_for_execute_engine", mutates_args=()
)
def no_op_placeholder_for_execute_engine(
    inputs: List[torch.Tensor],
    abi_version: str,
    name: str,
    serialized_device_info: str,
    serialized_engine: str,
    serialized_in_binding_names: str,
    serialized_out_binding_names: str,
    serialized_hardware_compatible: str,
    serialized_metadata: str,
    serialized_target_platform: str,
    serialized_require_output_allocator: str,
) -> List[torch.Tensor]:
    raise RuntimeError(
        "The saved model is cross compiled for windows in Linux, should only be loadded in Windows via torch_tensorrt.load_cross_compiled_exported_program() api."
    )
