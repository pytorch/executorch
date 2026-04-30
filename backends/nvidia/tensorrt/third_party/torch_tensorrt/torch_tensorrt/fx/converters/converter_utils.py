import operator
import warnings
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Target

from ..types import (
    Shape,
    TRTDataType,
    TRTElementWiseOp,
    TRTLayer,
    TRTNetwork,
    TRTPlugin,
    TRTPluginFieldCollection,
    TRTTensor,
)
from ..utils import Frameworks, unified_dtype_converter


class SourceIR(Enum):
    NN = auto()
    ACC = auto()
    ATEN = auto()
    PRIM = auto()
    UNKNOWN = auto()

    def __str__(self):
        if self == SourceIR.NN:
            return "nn"
        elif self == SourceIR.ACC:
            return "acc"
        elif self == SourceIR.ATEN:
            return "aten"
        elif self == SourceIR.PRIM:
            return "prim"
        else:
            return "unknown_ir"


def get_trt_plugin(
    plugin_name: str,
    field_collection: List[TRTPluginFieldCollection],
    version: str,
    plugin_namespace: str = "",
) -> TRTPlugin:
    """
    Get a TensorRT plugin based on the given parameters.

    Args:
        plugin_name (str): Name of the plugin.
        field_collection (List[TRTPluginFieldCollection]): Parameters that needed
            to create a plugin using the plugin creator.
        version (str): Version of the plugin.
        plugin_namespace (str): Namespace of the plugin.

    Returns:
        A TensorRT plugin that can be added to TensorRT network as Plugin layer.
    """
    # print the registered plugins
    # PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
    # for plugin_creator in PLUGIN_CREATORS:
    #     print(plugin_creator.name)

    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(
        plugin_name, version, plugin_namespace
    )
    assert plugin_creator, f"Unabled to find plugin creator with name {plugin_name}"
    plugin = plugin_creator.create_plugin(
        name=plugin_name, field_collection=field_collection
    )

    assert plugin is not None, f"Plugin: {plugin_name} could not be fetched"
    return plugin


def get_positive_dim(dim: int, dim_size: int) -> int:
    """
    Given an integer number that represents a dimension in the array,
    transform it to a positive integer dim if it's negative. Otherwise, do
    nothing.

    Args:
        dim (int): A integer number that represents a dimension in an array.
        dim_size (int): The size of the dimension in the array.

    Returns:
        A positive integer that represent the same dimension as the given dim.
    """
    if dim < 0:
        return dim % dim_size
    return dim


def set_layer_name(
    layer: TRTLayer,
    target: Union[Target, torch.nn.Module, str],
    name: str,
    source_ir: Optional[SourceIR] = None,
) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (TRTLayer): A TensorRT layer of which we want to set the name.
        target (Target): A fx node.target or submodule. For call_function node, it's the function that
            the node represents.
        name (str): Consists of fx node.name with optional suffix.
        source_ir: (Optional[SourceIR]): The IR producing the op.
    """

    source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN

    target_name = (
        f"{source_ir}_ops.{target}"
        if isinstance(target, str)
        else f"{source_ir}_ops.{target.__name__}"
    )
    layer.name = f"[{layer.type.name}]-[{target_name}]-[{name}]"


def extend_attr_to_tuple(
    val: Any,
    num_elem: int,
) -> Tuple[Any, ...]:
    """
    If `val` is not a tuple or a list, then we make a tuple of size `num_elem` by
    replicating `val` `num_elem` times.

    Args:
        val (Any): Value that we want to process.

    Returns:
        A tuple.
    """
    if not isinstance(val, (tuple, list)):
        val = (val,) * num_elem
    if isinstance(val, list):
        val = tuple(val)
    return val


def extend_mod_attr_to_tuple(mod: torch.nn.Module, name: str, size: int):
    """
    Extend an attribute of `mod` that named `name` to a tuple of `size`.
    """
    val = getattr(mod, name)
    return extend_attr_to_tuple(val, size)


def to_numpy(
    value: Optional[Union[torch.Tensor, np.ndarray, int, float]],
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]] = None,
) -> Optional[np.ndarray]:
    """
    Convert a PyTorch Tensor, Numpy array, or scalar to a Numpy Array. If the tensor is
    quantized it will be dequantized first.

    Args:
        value (Optional[Union[torch.Tensor, np.ndarray, int, float]]):
            A PyTorch tensor, Numpy array, int, or float
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.

    Returns:
        A Numpy array.
    """
    output = None

    if value is None:
        return None

    elif isinstance(value, np.ndarray):
        output = value

    elif isinstance(value, torch.Tensor):
        if value.is_quantized:
            value = value.dequantize()

        output = value.cpu().detach().contiguous().numpy()

    elif isinstance(value, int):
        output = np.array([value], dtype=np.int32)

    elif isinstance(value, float):
        output = np.array([value], dtype=np.float32)

    else:
        raise AssertionError(
            f"to_numpy can only be called on None, int, float, np.ndarray, or torch.Tensor, got: {value}"
        )

    return (
        output
        if dtype is None
        else output.astype(unified_dtype_converter(dtype, Frameworks.NUMPY))
    )


def has_dynamic_shape(shape: Shape) -> bool:
    """
    Determine if the given shape has dynamic dim. i.e. if there're -1 in shape.

    Args:
        shape (Shape): Shape of a tensor. Essentially is a sequence of integers.

    Returns:
        A boolean value indicates whether there's dynamic dim in the shape.
    """
    count = 0
    for s in shape:
        count += 1 if s == -1 else 0
    return count


def get_axes_for_reduce_op(
    dim: Union[int, Sequence[int]],
    has_implicit_batch_dimension: bool,
) -> int:
    """
    TensorRT reduce layer relies on the binary representation of axes to
    determine which dims to reduce. For example, if we want to reduce on
    dim 1 and 2 then axes should be 6(110).

    Args:
        dim (Union[int, Sequence[int]]): An integer or a sequence of integers
            that will be used to generate axes for TensorRT.
        has_implicit_batch_dimension (bool): Whether the TensorRT network is
            using implicit batch dimension.

    Returns:
        An integer which binary form can be used as axes for TensorRT reduce
        layer.
    """
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert 0 not in dim, "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


def create_constant(
    network: TRTNetwork,
    value: Union[int, float, np.ndarray, torch.Tensor],
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]],
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.

    Args:
        network (TRTNetwork): A TensorRT network to which we want to add
            a constant layer.
        value (Union[int, float, np.ndarray, torch.Tensor]): A literal value, Numpy array,
            or a PyTorch tensor that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    constant = network.add_constant(
        (1,) if isinstance(value, (int, float)) else value.shape,
        to_numpy(value, dtype),
    )
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(
    network: TRTNetwork,
    input_val: Any,
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]] = None,
) -> TRTTensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.

    Args:
        network (TRTNetwork): A TensorRT network. If we want to
            add a TensorRT Constant layer, we will add it to this network.
        input_val (Any): An value that we want to convert to a TensorRT ITensor.
        name (str): The name of the created TensorRT Constant layer if there's
            one.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If dtype is provided, the given value will be converted to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    # TRT can not add constant for bool type. We do a work around to 1) cast it to int and 2)cast to bool later
    # This is useful for logical operations which require input to be bool type
    if isinstance(input_val, bool):
        input_val = int(input_val)

    if isinstance(input_val, torch.Tensor) and (
        input_val.dtype == torch.bool or input_val.dtype == torch.int64
    ):
        input_val = input_val.to(torch.int32)
    elif isinstance(input_val, np.ndarray) and (
        input_val.dtype == np.bool_ or input_val.dtype == np.int64
    ):
        input_val = input_val.to(np.int32)

    if isinstance(input_val, (torch.Tensor, np.ndarray, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif isinstance(input_val, TRTTensor):
        return input_val

    raise RuntimeError(
        f"Received input {input_val} of name {name} that "
        "is not part of the TensorRT region!"
    )


def prepend_ones(
    network: TRTNetwork,
    tensor: TRTTensor,
    name: str,
    num_prepend_ones: int,
) -> TRTTensor:
    """
    Prepend 1s to the shape of TensorRT ITensor `tensor`.

    Args:
        network (TRTNetwork): The TensorRT network that `tensor`
            belongs to.
        tensor (TRTTensor): A TensorRT tensor.
        name (str): Name of the TensorRT Shuffle layer which is used to prepend
            1s.
        num_prepend_ones (int): Number of 1s that will be prepend.

    Returns:
        A Tensorrt ITensor which contains the same value as `tensor` but with
        more 1s prepended to the beginning of `tensor` shape.
    """
    layer = network.add_shuffle(tensor)

    # If there're dynamic dim in tensor's shape, we need to use shape layer to
    # compute the final shape.
    if has_dynamic_shape(tensor.shape):
        tensor_shape_layer = network.add_shape(tensor)
        tensor_shape = tensor_shape_layer.get_output(0)
        tensor_shape = type_cast(
            network, "shape", name + "shape_casted", tensor_shape, trt.int32
        )
        tensor_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), tensor_shape]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(tensor.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(
    network: TRTNetwork,
    a: TRTTensor,
    b: TRTTensor,
    a_name: str,
    b_name: str,
    preset_diff: int = 0,
) -> Tuple[TRTTensor, TRTTensor]:
    """
    Broadcast two TensorRT tensors to the same number of dimensions by
    prepending 1s to the tensor with less number of dimensions.

    Args:
        network (TRTNetwork): TensorRT network object.
        a (TRTTensor): A TensorRT ITensor.
        b (TRTTensor): A TensorRT ITensor.
        a_name (str): Name of tensor a.
        b_name (str): Name of tensor b.
        preset_diff (int): The difference of number of dimensions after broadcast.
            A positive number means after broadcast, tensor `a` would have `preset_diff`
            more dimensions than `b`. This is used in matmul, since we need to broadcast
            tensors but not always to the same number of dimension. The reason is that
            matmul supports Matrix x Vector and in this case broadcasted vector should
            have 1 less number of dimensions than the matrix tensor.

    Returns:
        Two TensorRT ITensors that are broadcasted to the same number of dimensions.
    """
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = prepend_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = prepend_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def get_shape_with_dynamic_shape(
    network: TRTNetwork,
    shape: Union[list, tuple, torch.Tensor],
    input_val: TRTTensor,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    Prepare the real output tensor shape for dynamic shape mode tensor input.
    How this functions works:
    Assuming the input_val has actual shape [2048, 256, 512], expected reduce operation
    output shape is [-1, 128, 256], this function should return [2048, 128, 256] as the actual
    reduce operation output shape. Steps of calculations are:
        1. get the actual tensor shape of input_val via add_shape layer;
        2. create a all 0 tensor [0, 0, 0];
        3. run elementwise comparision the [0, 0, 0] and [-1, 128, 256] tensor, get a condition tensor [True, False, False];
        4. use the condition tensor [True, False, False] to do selection between [2048, 256, 512] and [-1, 128, 256], replace
           all -1 dynamic shape dimensions with actual batch_size value;
        5. output shape with actual batch_size as [2048, 128, 256]

    Args:
        network (TRTNetwork): TensorRT network object.
        shape: calculated shape of the expected output tensor
        input_val (TRTTensor): A TensorRT ITensor.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
    Returns:
        TensorRT ITensors that represents the actual shape of the input_val
    """
    # Ger real shape info for input_val
    input_shape = network.add_shape(input_val).get_output(0)

    scale_layer = network.add_constant(
        input_shape.shape, np.ascontiguousarray(shape, dtype=np.int32)
    )
    set_layer_name(scale_layer, target, f"{name}_scale")
    scale_res = scale_layer.get_output(0)

    length = input_shape.shape[0]
    zero_layer = network.add_constant(
        input_shape.shape, to_numpy(torch.zeros((length), dtype=torch.int32))
    )
    set_layer_name(zero_layer, target, f"{name}_zeros")

    condition_val = add_binary_elementwise_layer(
        network,
        scale_res,
        zero_layer.get_output(0),
        trt.ElementWiseOperation.LESS,
        target,
        f"{name}_shape",
    )
    select_layer = network.add_select(condition_val, input_shape, scale_res)
    set_layer_name(select_layer, target, f"{name}_select")
    return select_layer.get_output(0)


def add_binary_elementwise_layer(
    network: TRTNetwork,
    lhs_val: Union[int, float, TRTTensor, torch.Tensor],
    rhs_val: Union[int, float, TRTTensor, torch.Tensor],
    op_type: trt.ElementWiseOperation,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    This function adds a TensorRT elementwise layer. We allow both operands to be
    constant (not a trt tensor) because in implicit batch dimension mode, we could
    introduce constant via .size() op. Other scenario should be const folded first.
    If any operand is not a trt tensor, we make it a trt constant layer while preserve
    its dtype. Then we broadcast these two inputs to have the same number of dimensions.

    Limitation:
        If we are using implicit batch dim mode, the operand that is not a trt
    tensor are not allowed to have larger ranks than the trt tensor operand.

    Args:
        network (TRTNetwork): TensorRT network object.
        lhs_val (TRTTensor): Left operand of the binary operation. Could
            be a TensorRT tensor, a PyTorch tensor or a simple value.
        rhs_val (TRTTensor): Right operand of the binary operation. Similar
            to lhs_val.
        op_type (trt.ElementWiseOperation): Type of the TensorRT elementwise binary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Elementwise layer.
    """
    lhs_dtype = None
    rhs_dtype = None
    is_lhs_trt_tensor = False
    is_rhs_trt_tensor = False

    if isinstance(lhs_val, TRTTensor):
        lhs_dtype = unified_dtype_converter(lhs_val.dtype, Frameworks.TORCH)
        is_lhs_trt_tensor = True
    if isinstance(rhs_val, TRTTensor):
        rhs_dtype = unified_dtype_converter(rhs_val.dtype, Frameworks.TORCH)
        is_rhs_trt_tensor = True

    if not is_lhs_trt_tensor and not is_rhs_trt_tensor:
        warnings.warn(
            f"Both operands of the binary elementwise op {name} "
            "are constant. In this case, please consider constant fold the model first."
        )
        return get_python_op_from_trt_elementwise_op(op_type)(lhs_val, rhs_val)

    # If the following conditions are true:
    #  1. the network has implicit batch dimension,
    #  2. one operand has shape [] (real shape is [batch_size]),
    #  3. another operand is a scalar,
    # then the result should also have shape [] (real shape is [batch_size]).
    #
    # In such case, we need to convert the scalar operand to tensor, because
    # this way the shape will become [1], and then will be properly squeezed
    # into [], meaning that the result will have shape [], which is what we
    # expect.
    #
    # Note that the dtype here is supposed to be the same as the scalar
    # dtype but we don't have a way to detect whether it makes sense for the
    # scalar to be float or half. Hence we go with the lhs dtype.
    if is_lhs_trt_tensor and isinstance(rhs_val, (float, int)):
        rhs_val = np.array(
            [rhs_val], dtype=unified_dtype_converter(lhs_val.dtype, Frameworks.NUMPY)
        )
    if is_rhs_trt_tensor and isinstance(lhs_val, (float, int)):
        lhs_val = np.array(
            [lhs_val], dtype=unified_dtype_converter(rhs_val.dtype, Frameworks.NUMPY)
        )

    # When lhs is scalar, and rhs has shape [1,], then currently the assert
    # will fail because lhs shape has fewer dimensions than rhs shape.  This
    # happens when using implicit batch dimension, when we removed the 1st
    # dimension from input tensor, causing it to have shape [] - a scalar.  We
    # fix it by reducing the rhs constant with a squeeze_left, so it becomes a
    # scalar too. More generally, we squeeze_left on input if it's a constant
    # tensor. This is safe because broadcast will pad dimensions on the left
    # (prepend) to make lhs and rhs shape compatible.
    if network.has_implicit_batch_dimension:
        if isinstance(lhs_val, (torch.Tensor, np.ndarray)):
            lhs_val = squeeze_left(lhs_val)
        if isinstance(rhs_val, (torch.Tensor, np.ndarray)):
            rhs_val = squeeze_left(rhs_val)

    lhs_val = get_trt_tensor(network, lhs_val, f"{name}_lhs", lhs_dtype)
    rhs_val = get_trt_tensor(network, rhs_val, f"{name}_rhs", rhs_dtype)

    # Check the limitation in the doc string.
    if network.has_implicit_batch_dimension:
        if is_lhs_trt_tensor and not is_rhs_trt_tensor:
            assert len(lhs_val.shape) >= len(
                rhs_val.shape
            ), f"{lhs_val.shape} >= {rhs_val.shape}"
        elif not is_lhs_trt_tensor and is_rhs_trt_tensor:
            assert len(rhs_val.shape) >= len(
                lhs_val.shape
            ), f"{rhs_val.shape} >= {lhs_val.shape}"

    lhs_val, rhs_val = broadcast(
        network, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs"
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    set_layer_name(layer, target, name)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return output


def squeeze_left(const: Union[torch.Tensor, np.ndarray]):
    """
    Squeeze the size-1 dimensions on the left side of the shape tuple.
    PyTorch's `squeeze()` doesn't support passing multiple `dim`s at once, so
    we do it iteratively.
    """
    while len(const.shape) > 0 and const.shape[0] == 1:
        if isinstance(const, torch.Tensor):
            const = const.squeeze(dim=0)
        elif isinstance(const, np.ndarray):
            const = const.squeeze(axis=0)
        else:
            raise AssertionError(f"Expected torch Tensor or Numpy array, got: {const}")
    return const


def add_unary_layer(
    network: TRTNetwork,
    input_val: TRTTensor,
    operation_type: trt.UnaryOperation,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return layer.get_output(0)


def add_reduce_layer(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    operation_type: trt.ActivationType,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Reduce layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        target (Target): Target of fx node.
        args (Tuple[Argument, ...]): Args of the fx node.
        kwargs (Dict[str, Argument]): Kwargs of the fx node.
        operation_type (trt.ElementWiseOperation): Type of the TensorRT activation
            operation.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Reduce layer.
    """
    input_val = kwargs["input"]
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # If dim is specified, then the op is reducing over certain dimensions.
    # Otherwise, it's reducing over all elements, which is only supported in
    # explicit batch dimension.
    if "dim" not in kwargs:
        assert (
            not network.has_implicit_batch_dimension
        ), f"We don't support reduce({name}) over all the elements if batch dim is implicit."
        dim = range(0, len(input_val.shape))
    else:
        dim = kwargs["dim"]  # type: ignore[assignment]

    if not isinstance(dim, Sequence):
        dim = (dim,)

    if not network.has_implicit_batch_dimension:
        dim = tuple(len(input_val.shape) + i if i < 0 else i for i in dim)
    else:
        dim = tuple(len(input_val.shape) + i + 1 if i < 0 else i for i in dim)

    keepdim = False if "keepdim" not in kwargs else kwargs["keepdim"]
    layer = network.add_reduce(
        input_val,
        operation_type,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        keepdim,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def get_dyn_range(scale, zero_point, dtype):
    """
    Get the dynamic range of a tensor based on its scale, zero_point and dtype.
    """
    if dtype == torch.quint8:
        min_val, max_val = 0, 255
    elif dtype == torch.qint8:
        min_val, max_val = -128, 127
    else:
        raise RuntimeError(f"Unsupported quantized dtype {dtype}")

    return (min_val - zero_point) * scale, (max_val - zero_point) * scale


def mark_as_int8_layer(layer, dynamic_range):
    """
    Set the precision of a layer to int8 as well as the type of its first output.
    Also set the dynamic range of its first output.
    """
    if layer.type not in {
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONCATENATION,
        trt.LayerType.CONSTANT,
        trt.LayerType.SHAPE,
    }:
        layer.precision = trt.int8

    for i in range(layer.num_outputs):
        output_val = layer.get_output(i)
        output_val.dynamic_range = dynamic_range
        layer.set_output_type(i, trt.int8)
        # output_val.dtype = trt.int8


def get_inputs_from_args_and_kwargs(args, kwargs, input_names):
    inputs = []
    for i, key in enumerate(input_names):
        if key not in kwargs:
            inputs.append(args[i])
        else:
            inputs.append(kwargs[key])
    return inputs


def sign(
    network: TRTNetwork, input_val: TRTTensor, target: Target, name: str
) -> TRTTensor:
    """
    Sign is calculated as below:
       x = input
       sign = (exp(x) // exp(abs(x))) * 2 - 1
       For positive number and 0, (exp(x) // exp(abs(x))) yield 1; for negative number, (exp(x) // exp(abs(x))) yield 0.
       With multiply 2, the value become 2(for pos and 0) and 0(for neg).
       Finally minus 1, the value become 1(for pos and 0) and -1(for neg).

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): The input tensor.
        target (Target): fx node target.
        name (str): Name of the fx node with optional suffix.

    Returns:
        A TensorRT tensor represent the result of sign operator.
    """
    input_exp_output = add_unary_layer(
        network, input_val, trt.UnaryOperation.EXP, target, f"{name}_prod_exp"
    )
    input_abs_output = add_unary_layer(
        network, input_val, trt.UnaryOperation.ABS, target, f"{name}_prod_abs"
    )
    input_abs_exp_output = add_unary_layer(
        network,
        input_abs_output,
        trt.UnaryOperation.EXP,
        target,
        f"{name}_prod_abs_exp",
    )
    floor_div_output = add_binary_elementwise_layer(
        network,
        input_exp_output,
        input_abs_exp_output,
        trt.ElementWiseOperation.FLOOR_DIV,
        target,
        f"{name}_exp_floor_div",
    )
    double_floor_div_output = add_binary_elementwise_layer(
        network,
        floor_div_output,
        2,
        trt.ElementWiseOperation.PROD,
        target,
        f"{name}_floor_div*2",
    )
    return add_binary_elementwise_layer(
        network,
        double_floor_div_output,
        1,
        trt.ElementWiseOperation.SUB,
        target,
        f"{name}_sign",
    )


def trunc_div(
    input: TRTTensor, other: TRTTensor, network: TRTNetwork, target: Target, name: str
) -> TRTTensor:
    """
    Perform trunc divide on Tensor, result of divide will be round toward zero.
    This means for positive number, it will be floor round; for negative number,
    it will be ceil round. Example: [2.1, 0.8, -3.2] -> [2, 0, -3].

    Args:
        input: divisor.
        other: dividend.
        network: INetworkDefinition.
        target: node target.
        name: namespace for the op

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = add_binary_elementwise_layer(
        network, input, other, trt.ElementWiseOperation.PROD, target, f"{name}_prod"
    )
    sign_output = sign(network, prod_output, target, name)

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(network, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            network,
            other,
            f"{name}_other",
            dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH),
        )

    abs_input_output = add_unary_layer(
        network, input, trt.UnaryOperation.ABS, target, f"{name}_abs_input"
    )
    abs_other_output = add_unary_layer(
        network, other, trt.UnaryOperation.ABS, target, f"{name}_abs_other"
    )
    abs_floor_output = add_binary_elementwise_layer(
        network,
        abs_input_output,
        abs_other_output,
        trt.ElementWiseOperation.FLOOR_DIV,
        target,
        f"{name}_floor_div",
    )
    output = add_binary_elementwise_layer(
        network,
        abs_floor_output,
        sign_output,
        trt.ElementWiseOperation.PROD,
        target,
        f"{name}_output",
    )

    return output


def get_python_op_from_trt_elementwise_op(
    trt_op: TRTElementWiseOp,
) -> Callable[[Any, Any], Any]:
    if trt_op == trt.ElementWiseOperation.SUM:
        return operator.add
    elif trt_op == trt.ElementWiseOperation.PROD:
        return operator.mul
    elif trt_op == trt.ElementWiseOperation.SUB:
        return operator.sub
    elif trt_op == trt.ElementWiseOperation.DIV:
        return operator.truediv
    elif trt_op == trt.ElementWiseOperation.FLOOR_DIV:
        return operator.floordiv
    else:
        raise RuntimeError(f"{trt_op} is not supported yet!")


def dtype_uniform(
    network: TRTNetwork, target: Target, name: str, input: TRTTensor, other: TRTTensor
):
    table = {trt.bool: 0, trt.int32: 1, trt.float16: 2, trt.float32: 3}
    input_dtype = input.dtype
    other_dtype = other.dtype
    if table[input_dtype] > table[other_dtype]:
        layer = network.add_identity(other)
        layer.set_output_type(0, input_dtype)
        set_layer_name(layer, target, f"{name}_other_dtype_change")
        other = layer.get_output(0)
    elif table[input_dtype] < table[other_dtype]:
        layer = network.add_identity(input)
        layer.set_output_type(0, other_dtype)
        set_layer_name(layer, target, f"{name}_input_dtype_change")
        input = layer.get_output(0)
    elif table[input_dtype] == 0 and table[other_dtype] == 0:
        layer_i = network.add_identity(input)
        layer_i.set_output_type(0, trt.int32)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input = layer_i.get_output(0)

        layer_o = network.add_identity(other)
        layer_o.set_output_type(0, trt.int32)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other = layer_o.get_output(0)
    return input, other


def type_cast(
    network: TRTNetwork,
    target: Target,
    name: str,
    input: TRTTensor,
    cast_type: TRTDataType,
):
    """
    This function helps to cast the input type to cast_type
    """
    layer_i = network.add_cast(input, cast_type)
    set_layer_name(layer_i, target, f"{name}_dtype_change")
    return layer_i.get_output(0)
