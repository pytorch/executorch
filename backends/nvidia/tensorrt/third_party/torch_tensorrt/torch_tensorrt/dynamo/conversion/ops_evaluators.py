# mypy: disallow-untyped-decorators=False

import logging
import operator
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.utils import (
    Frameworks,
    unified_dtype_converter,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


def getitem_validator(getitem_node: Node, settings: CompilationSettings = None) -> bool:
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

    # Getitem nodes can only be converted if their parent node also can
    return (
        getitem_node.args[0] in DYNAMO_CONVERTERS
        or getitem_node.args[0].op == "get_attr"
    )


# TODO: Subsequent evaluators should be registered here with their own validators
@dynamo_tensorrt_converter(
    operator.getitem,
    capability_validator=getitem_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(torch.ops.aten.detach.default, supports_dynamic_shapes=True)
def generic_evaluator(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    _LOGGER.debug(
        f"Evaluating {ConverterRegistry.qualified_name_or_str(target)} on object with name: {name}"
    )
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        return target(*args)


def rand_validator(rand_node: Node, settings: CompilationSettings = None) -> bool:
    dtype = rand_node.kwargs.get("dtype", None)
    layout = rand_node.kwargs.get("layout", None)
    if dtype is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying output dtype, got {dtype}."
        )
        return False
    if layout is not None:
        _LOGGER.debug(f"Currently we don't support specifying layout, got {layout}.")
        return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.rand.default, capability_validator=rand_validator
)
def aten_ops_rand(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return np.random.rand(*args[0])


@dynamo_tensorrt_converter(
    torch.ops.aten.randn.default, capability_validator=rand_validator
)
def aten_ops_randn(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return np.random.randn(*args[0])


def randperm_validator(
    randperm_node: Node, settings: CompilationSettings = None
) -> bool:
    dtype = randperm_node.kwargs.get("dtype", None)
    layout = randperm_node.kwargs.get("layout", None)
    input = randperm_node.args[0]
    if not isinstance(input, int):
        _LOGGER.error("Input should be of type int.")
        return False
    if dtype is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying output dtype, got {dtype}."
        )
        return False
    if layout is not None:
        _LOGGER.debug(f"Currently we don't support specifying layout, got {layout}.")
        return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.randperm.default, capability_validator=randperm_validator
)
def aten_ops_randperm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return np.random.permutation(args[0])


def empty_validator(empty_node: Node, settings: CompilationSettings = None) -> bool:
    device = empty_node.kwargs.get("device", None)
    if device is not None:
        _LOGGER.debug(f"Currently we don't support specifying device, got {device}.")
        return False
    layout = empty_node.kwargs.get("layout", None)
    if layout is not None:
        _LOGGER.debug(f"Currently we don't support specifying layout, got {layout}.")
        return False
    memory_format = empty_node.kwargs.get("memory_format", None)
    if memory_format is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying memory_format, got {memory_format}."
        )
        return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.empty.memory_format, capability_validator=empty_validator
)
def aten_ops_empty(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    empty_np_tensor = None
    if kwargs.get("dtype") is not None:
        empty_np_tensor = np.empty(
            tuple(args[0]),
            dtype=unified_dtype_converter(kwargs.get("dtype"), Frameworks.NUMPY),
        )
    else:
        # default returns np.float64. Verify the correctness of this
        empty_np_tensor = np.empty(tuple(args[0]))
    return empty_np_tensor
