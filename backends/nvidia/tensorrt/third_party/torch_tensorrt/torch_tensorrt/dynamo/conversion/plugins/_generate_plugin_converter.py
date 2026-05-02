import logging
import uuid
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt._features import needs_qdp_plugin
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    ConverterPriority,
    DynamoConverterImplSignature,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def _generate_plugin_converter(
    namespace: str,
    op_name: str,
    overload: Optional[str] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    use_aot_if_available: bool = True,
) -> DynamoConverterImplSignature:
    try:
        import tensorrt.plugin as trtp

    except ImportError as e:
        raise RuntimeError(
            "Unable to import TensorRT plugin. TensorRT version must be 10.7.0 or higher to support for Triton based TensorRT plugins"
        )
    from tensorrt.plugin._lib import QDP_REGISTRY

    torch_target = getattr(getattr(torch.ops, namespace), op_name)
    overload_str = overload if overload else ""
    overload_name = overload_str if overload else "default"
    torch_overload = getattr(torch_target, overload_name)
    assert (
        f"{namespace}::{op_name}" in QDP_REGISTRY
    ), f"Could not find a tensorrt plugin registered for op {namespace}::{op_name}, unable to generate converter"
    torch_schema = torch_target._schemas[overload_str]

    use_aot_plugin = use_aot_if_available

    if use_aot_if_available:
        desc = QDP_REGISTRY[f"{namespace}::{op_name}"]
        if desc.aot_impl_func is None:
            use_aot_plugin = False
            _LOGGER.debug(
                f"AOT impl func not found for {namespace}::{op_name}, use JIT plugin instead"
            )

    def custom_kernel_converter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:

        plugin = getattr(getattr(trtp.op, namespace), op_name)

        tensor_inputs = plugin.input_tensor_names
        tensor_args = args[0 : len(tensor_inputs)]

        unique_id = uuid.uuid4()
        itensor_args = [
            get_trt_tensor(ctx, t, f"{t_name}_{unique_id}")
            for (t, t_name) in zip(tensor_args, tensor_inputs)
        ]

        # Assuming TensorRT preserves kwargs order like PyTorch does
        non_tensor_inputs = plugin.input_attrs

        kwargs = {}

        for arg in torch_schema.arguments:
            if arg.default_value is not None:
                kwargs[arg.name] = arg.default_value

        non_tensor_args = args[len(tensor_inputs) :]
        non_tensor_kwargs = dict(zip(list(non_tensor_inputs.keys()), non_tensor_args))

        # Update kwargs with non_tensor_kwargs, adding new keys or overwriting existing ones
        kwargs.update(non_tensor_kwargs)

        for k, v in kwargs.items():
            if isinstance(v, torch.fx.immutable_collections.immutable_list):
                kwargs[k] = np.array(v)

        layer = ctx.net.add_plugin(plugin(*itensor_args, **kwargs), aot=use_aot_plugin)
        assert layer, f"{namespace}::{name} plugin layer was not able to be created"
        _LOGGER.debug(
            f"Adding generated plugin for {namespace}::{name} to tensorrt network"
        )
        layer.name = f"[{target}]-[{name}]"
        return layer.get_output(0)

    custom_kernel_converter = dynamo_tensorrt_converter(
        torch_overload,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
    )(custom_kernel_converter)
    assert (
        torch_overload in DYNAMO_CONVERTERS
    ), f"Generated dynamo converter for {namespace}::{op_name} did not get properly registered in the converter registry"
    return custom_kernel_converter


@needs_qdp_plugin
def generate_plugin_converter(
    plugin_id: str,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    use_aot_if_available: bool = True,
) -> DynamoConverterImplSignature:
    plugin_ns, plugin_name = plugin_id.split("::")
    return _generate_plugin_converter(
        plugin_ns,
        plugin_name,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        use_aot_if_available=use_aot_if_available,
    )
