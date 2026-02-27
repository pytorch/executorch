# encoding: utf-8
import logging
import operator
import warnings

import torch  # isort:skip
from typing import cast, Iterable, List, Optional, Sequence

import torch.nn as nn
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

from . import acc_utils
from .acc_normalizer import (
    register_acc_op,
    register_acc_op_mapping,
    register_custom_acc_mapper_fn,
)
from .acc_op_properties import AccOpProperty, register_acc_op_properties

logger: logging.Logger = logging.getLogger(__name__)

this_arg_is_optional = True
move_to_qparams = True
dont_move_to_qparams = False

# A proxy embedding size. We use this for tracing proxy operators using XL
# weights which we can't load into memory (because they're too large), we
# instead substitute a smaller weight with embedding size =
# PROXY_EMBEDDING_SIZE.
PROXY_EMBEDDING_SIZE = 8


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.linear))
@register_acc_op
def linear(*, input, weight, bias):
    return nn.functional.linear(input=input, weight=weight, bias=bias)


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op
def quantized_linear(*, input, weight, bias, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    return nn.quantized.functional.linear(
        input,
        weight,
        bias,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "flatten"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("start_dim", "start_dim", this_arg_is_optional),
        ("end_dim", "end_dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.flatten))
@register_acc_op
def flatten(*, input, start_dim=0, end_dim=-1):
    return torch.flatten(input=input, start_dim=start_dim, end_dim=end_dim)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "squeeze"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.squeeze),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def squeeze(*, input, dim=None):
    if dim is None:
        return input.squeeze()
    return input.squeeze(dim=dim)


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.embedding))
@register_acc_op
def embedding(
    *,
    input,
    weight,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
):
    return torch.nn.functional.embedding(**locals())


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.max_pool1d))
@register_acc_op
def max_pool1d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    return_indices,
):
    return nn.functional.max_pool1d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.max_pool2d))
@register_acc_op
def max_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    return_indices,
):
    return nn.functional.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.max_pool3d))
@register_acc_op
def max_pool3d(
    *, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
):
    return nn.functional.max_pool3d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.normalize))
@register_acc_op
def normalize(*, input, p, dim, eps, out):
    return nn.functional.normalize(
        input=input,
        p=p,
        dim=dim,
        eps=eps,
        out=out,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.adaptive_avg_pool2d)
)
@register_acc_op
def adaptive_avg_pool2d(*, input, output_size):
    return nn.functional.adaptive_avg_pool2d(input=input, output_size=output_size)


@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.adaptive_avg_pool3d)
)
@register_acc_op
def adaptive_avg_pool3d(*, input, output_size):
    return nn.functional.adaptive_avg_pool3d(input=input, output_size=output_size)


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.avg_pool1d))
@register_acc_op
def avg_pool1d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
):
    return nn.functional.avg_pool1d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.avg_pool2d))
@register_acc_op
def avg_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return nn.functional.avg_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.avg_pool3d))
@register_acc_op
def avg_pool3d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return nn.functional.avg_pool3d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sign))
@register_acc_op
def sign(*, input):
    return torch.sign(input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "type"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def custom_type_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_obj = node.kwargs["input"]
    dtype_obj = node.kwargs.get("dtype")
    with node.graph.inserting_before(node):
        if dtype_obj is None:
            dtype_node = node.graph.call_function(dtype, kwargs={"input": input_obj})
            dtype_node.meta["type"] = torch.dtype
            return dtype_node
        else:
            new_kwargs = {
                "input": input_obj,
                "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=dtype_obj),
            }
            new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
            new_node.meta = node.meta
            return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "type_as"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("tensor", "tensor"),
    ],
)
def custom_type_as_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_obj = node.kwargs["input"]
    other_obj = node.kwargs["tensor"]
    with node.graph.inserting_before(node):
        dtype_node = node.graph.call_function(dtype, kwargs={"input": other_obj})
        dtype_node.meta["type"] = torch.dtype
        device_node = node.graph.call_function(device, kwargs={"input": other_obj})
        device_node.meta["type"] = torch.device

        new_kwargs = {
            "input": input_obj,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=dtype_node),
            "device": device_node,
        }
        new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
        new_node.meta = node.meta
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def dtype(*, input):
    return input.dtype


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def size(*, input):
    return input.size()


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def device(*, input):
    return input.device


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.numel))
@register_acc_op
def numel(*, input):
    return torch.numel(input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", getattr),
    arg_replacement_tuples=[],
)
def custom_getattr_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Custom function for mapping a call_function getattr to other ops.

    Supports:
    * getattr on a torch.Tensor with "shape", "device", or "dtype" attributes
    * getattr for accessing named tuples
    """
    # Have to use args here since getattr forces positional args.
    input_obj = node.args[0]
    attr_name = node.args[1]
    assert isinstance(input_obj, torch.fx.Node)
    input_obj_type = input_obj.meta["type"]

    # Handle named tuple access. NamedTupleMeta and the namedtuple factory function
    # create a subclass of tuple with an extra _fields attribute.
    if issubclass(input_obj_type, tuple) and hasattr(input_obj_type, "_fields"):
        idx = None
        for i, name in enumerate(input_obj_type._fields):
            if name == attr_name:
                idx = i
                break
        assert (
            idx is not None
        ), f"Named tuple type {input_obj_type} does not have field {name}"

        with node.graph.inserting_before(node):
            getitem_node = node.graph.call_function(
                getitem, kwargs={"input": input_obj, "idx": idx}
            )
            getitem_node.meta = node.meta.copy()
            return getitem_node

    assert input_obj_type in [
        torch.Tensor,
        torch.nn.parameter.Parameter,
    ], f"Expected torch.Tensor type for {input_obj_type}"
    assert (
        attr_name == "shape" or attr_name == "device" or attr_name == "dtype"
    ), f"Only supporting shape, device and dtype getattr for now, not {attr_name}"
    if attr_name == "shape":
        func = size
    elif attr_name == "device":
        func = device
    elif attr_name == "dtype":
        func = dtype
    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(func, kwargs={"input": input_obj})
        size_node.meta = node.meta.copy()
        return size_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "size"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
def tensor_size_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from Tensor.size() to acc_ops.size. We map size() to acc_ops.size directly
    and map size(dim) to acc_ops.size + acc_ops.getitem.
    """

    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(
            size, kwargs={"input": node.kwargs["input"]}
        )

        if "dim" not in node.kwargs:
            size_node.meta = node.meta.copy()
            return size_node

        size_node.meta["type"] = torch.Size
        getitem_node = node.graph.call_function(
            getitem, kwargs={"input": size_node, "idx": node.kwargs["dim"]}
        )
        getitem_node.meta = node.meta.copy()
        return getitem_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.add))
@register_acc_op_mapping(op_and_target=("call_method", "add"))
@register_acc_op
def add(*, input, other):
    if not (isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor)):
        return operator.add(input, other)
    else:
        return input + other


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "unsqueeze"))
@register_acc_op_mapping(op_and_target=("call_function", torch.unsqueeze))
@register_acc_op
def unsqueeze(*, input, dim: int):
    return torch.unsqueeze(input=input, dim=dim)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "tile"))
@register_acc_op_mapping(op_and_target=("call_function", torch.tile))
@register_acc_op
def tile(*, input, dims):
    return torch.tile(input=input, dims=dims)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "repeat"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "sizes"),
    ],
    skip_normalization_if_none=True,
)
def repeat_mapper(node: torch.fx.Node, _: nn.Module) -> Optional[torch.fx.Node]:
    """
    Map repeat to tile.
    """
    with node.graph.inserting_before(node):
        inputs = node.kwargs["input"]
        dims = node.kwargs["sizes"]
        # Skip repeat mapping when the list of dims is not all ints (ie. contains
        # some calculated value). torch.tile cannot support cases where dims
        # are Proxy nodes
        if (
            isinstance(dims, (list, tuple))
            and len(dims) > 0
            and not all(isinstance(x, int) for x in dims)
        ):
            logger.info(
                "Not mapping repeat to an acc op. We can't handle variable dims."
            )
            return
        new_node = node.graph.create_node(
            "call_function",
            tile,
            kwargs={"input": inputs, "dims": dims},
            name=f"{node.name}_repeat_map",
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "repeat_interleave"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("repeats", "repeats"),
        ("dim", "dim", this_arg_is_optional),
        ("output_size", "output_size", this_arg_is_optional),
    ],
    skip_normalization_if_none=True,
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.repeat_interleave),
    arg_replacement_tuples=[
        ("input", "input"),
        ("repeats", "repeats"),
        ("dim", "dim", this_arg_is_optional),
        ("output_size", "output_size", this_arg_is_optional),
    ],
    skip_normalization_if_none=True,
)
def repeat_interleave_mapper(node: torch.fx.Node, _: nn.Module):
    input_node = node.kwargs["input"]
    repeats = cast(int, node.kwargs["repeats"])
    dim = node.kwargs["dim"]
    if not (type(repeats) is int):
        logger.info(
            "Not mapping repeat_interleave to an acc op. We currently only support `repeat_interleave` with int repeats"
        )
        return
    rank = node.meta["tensor_rank"]
    if dim is None:
        repeat_dim = rank - 1
    else:
        assert type(dim) is int, "dim should be an int"
        repeat_dim = dim
    tile_dims = [1] * (rank + 1)
    tile_dims[repeat_dim + 1] = repeats

    with node.graph.inserting_before(node):
        unsqueeze_node = node.graph.create_node(
            "call_function",
            unsqueeze,
            kwargs={"input": input_node, "dim": repeat_dim + 1},
            name=f"{node.name}_unsqueeze",
        )
        tile_node = node.graph.create_node(
            "call_function",
            tile,
            kwargs={"input": unsqueeze_node, "dims": tuple(tile_dims)},
            name=f"{node.name}_repeat_interleave_map_tile",
        )
        new_shape = []
        if dim is not None:
            if dim < 0:
                repeat_dim = dim + rank
            else:
                repeat_dim = dim
            size_node = node.graph.create_node(
                "call_function",
                size,
                kwargs={"input": input_node},
                name=f"{node.name}_size",
            )
            size_node.meta["type"] = torch.Size
            for i in range(rank):
                shape_i = node.graph.create_node(
                    "call_function",
                    getitem,
                    kwargs={"input": size_node, "idx": i},
                    name=f"{node.name}_size_{i}",
                )
                if i == repeat_dim:
                    new_shape.append(-1)
                else:
                    new_shape.append(shape_i)
        else:
            new_shape.append(-1)

        reshaped_node = node.graph.create_node(
            "call_function",
            reshape,
            kwargs={
                "input": tile_node,
                "acc_out_ty": acc_utils.build_raw_tensor_meta(shape=new_shape),
            },
            name=f"{node.name}_reshape",
        )
        reshaped_node.meta = node.meta.copy()
        return reshaped_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.stack),
    arg_replacement_tuples=[
        ("tensors", "tensors"),
        ("dim", "dim"),
    ],
)
def stack_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Map torch.stack to unsqueeze + cat.
    """
    with node.graph.inserting_before(node):
        inputs = node.kwargs["tensors"]
        unsqueeze_nodes = []
        assert isinstance(inputs, Sequence)
        for i, t in enumerate(inputs):
            new_node = node.graph.create_node(
                "call_function",
                unsqueeze,
                kwargs={"input": t, "dim": node.kwargs["dim"]},
                name=f"{node.name}_unsqueeze_{i}",
            )
            new_node.meta["type"] = torch.Tensor
            unsqueeze_nodes.append(new_node)
        cat_node = node.graph.create_node(
            "call_function",
            cat,
            kwargs={"tensors": unsqueeze_nodes, "dim": node.kwargs["dim"]},
        )
        cat_node.meta = node.meta.copy()
        return cat_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.clamp))
@register_acc_op_mapping(op_and_target=("call_function", torch.clip))
@register_acc_op_mapping(op_and_target=("call_method", "clamp"))
@register_acc_op_mapping(op_and_target=("call_method", "clip"))
@register_acc_op
def clamp(*, input, min=None, max=None):
    return torch.clamp(input=input, min=min, max=max)


@register_acc_op_mapping(op_and_target=("call_function", torch.concat))
@register_acc_op_mapping(op_and_target=("call_function", torch.cat))
@register_acc_op
def cat(*, tensors, dim):
    return torch.cat(tensors=tensors, dim=dim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.transpose),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "transpose"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
def transpose_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    # Get the dim-permutation/shuffle
    ranks = node.meta["tensor_rank"]
    shuffle = list(range(ranks))
    dim0 = cast(int, node.kwargs["dim0"])
    dim1 = cast(int, node.kwargs["dim1"])
    shuffle[dim0] = dim1
    shuffle[dim1] = dim0

    # Create the new acc_ops.permute node. Update all uses of the transpose
    # node and then delete the transpose node.
    with node.graph.inserting_after(node):
        permute_node = node.graph.call_function(
            the_function=permute,
            kwargs={
                "input": node.kwargs.get("input"),
                "permutation": shuffle,
            },
        )
        permute_node.meta = node.meta.copy()
        node.replace_all_uses_with(permute_node)

    permute_node.graph.erase_node(node)
    return permute_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "contiguous"))
@register_acc_op
def contiguous(*, input):
    return input.contiguous()


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "softmax"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.softmax),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def softmax(*, input, dim, dtype=None):
    """
    _stacklevel are ignored here.
    """
    return torch.nn.functional.softmax(input=input, dim=dim, dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.addmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat1", "mat1"),
        ("mat2", "mat2"),
        ("beta", "beta"),
        ("alpha", "alpha"),
    ],
)
def addmm_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from torch.addmm to acc_ops.mm -> acc_ops.add, if alpha or beta is not 1
    then we also insert acc_ops.mul to the right place.
    """
    with node.graph.inserting_before(node):
        mm_kwargs = {"input": node.kwargs["mat1"], "other": node.kwargs["mat2"]}
        mm_node = node.graph.create_node(
            "call_function", matmul, kwargs=mm_kwargs, name=f"{node.name}_mm"
        )
        mm_node.meta = node.meta.copy()

        if node.kwargs["alpha"] != 1:
            mul_kwargs = {"input": mm_node, "other": node.kwargs["alpha"]}
            mm_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{mm_node.name}_mul"
            )
        mm_node.meta = node.meta.copy()

        input_node = node.kwargs["input"]
        if node.kwargs["beta"] != 1:
            mul_kwargs = {"input": input_node, "other": node.kwargs["beta"]}
            new_input_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{node.name}_input_mul"
            )
            assert isinstance(input_node, torch.fx.Node)
            new_input_node.meta = input_node.meta.copy()
            input_node = new_input_node

        add_kwargs = {"input": mm_node, "other": input_node}
        add_node = node.graph.create_node(
            "call_function", add, kwargs=add_kwargs, name=f"{node.name}_add"
        )
        add_node.meta = node.meta.copy()
        return add_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.t),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "t"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def t_mapper(node: torch.fx.Node, _: nn.Module):
    ranks = node.meta["tensor_rank"]
    shuffle = [1, 0] if (ranks > 1) else [0]

    with node.graph.inserting_before(node):
        new_node = node.graph.create_node(
            "call_function",
            permute,
            kwargs={"input": node.kwargs["input"], "permutation": shuffle},
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "permute"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "permutation"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.permute),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dims", "permutation"),
    ],
)
@register_acc_op
def permute(*, input, permutation):
    return input.permute(*permutation)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.square),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def square_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            mul, kwargs={"input": input_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_mapping(
    op_and_target=("call_method", "mm"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "matmul"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", operator.matmul),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.bmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.mm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.matmul))
@register_acc_op
def matmul(*, input, other):
    return torch.matmul(input=input, other=other)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout1d),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout2d),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout3d),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "detach"), arg_replacement_tuples=[("input", "input")]
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.detach),
    arg_replacement_tuples=[("input", "input")],
)
def dropout_mapper(node: torch.fx.Node, mod: nn.Module):
    """
    Remove dropout node and directly map its input to output.
    """
    return node.kwargs["input"]


try:
    from torchvision.ops import stochastic_depth

    assert callable(stochastic_depth)
except Exception as e:
    warnings.warn(
        f"Unable to import torchvision related libraries.: {e}. Please install torchvision lib in order to lower stochastic_depth"
    )
else:

    @register_custom_acc_mapper_fn(
        op_and_target=("call_function", stochastic_depth),
        arg_replacement_tuples=[("input", "input")],
    )
    def stochastic_depth_mapper(node: torch.fx.Node, mod: nn.Module):
        """
        Remove dropout node and directly map its input to output.
        """
        return node.kwargs["input"]


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.hardtanh),
)
@register_acc_op
def hardtanh(*, input, min_val=-1.0, max_val=1.0):
    return nn.functional.hardtanh(input=input, min_val=min_val, max_val=max_val)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", nn.functional.hardsigmoid))
@register_acc_op
def hardsigmoid(*, input):
    return nn.functional.hardsigmoid(input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.silu),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def silu(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        sigmoid_node = node.graph.call_function(sigmoid, kwargs={"input": input_node})
        sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(
            mul, kwargs={"input": sigmoid_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.hardswish),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def hardswish_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_sigmoid_node = node.graph.call_function(
            hardsigmoid, kwargs={"input": input_node}
        )
        new_sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(
            mul, kwargs={"input": new_sigmoid_node, "other": input_node}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.add),
    arg_replacement_tuples=[
        ("qa", "input"),
        ("qb", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_add(*, input, other, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    return torch.ops.quantized.add(
        input,
        other,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.mul),
    arg_replacement_tuples=[
        ("qa", "input"),
        ("qb", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_mul(*, input, other, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    return torch.ops.quantized.mul(
        input,
        other,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_tensor),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
        ("dtype", "dtype"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
        ("dtype", "dtype", dont_move_to_qparams),
    ],
)
@register_acc_op
def quantize_per_tensor(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    dtype = acc_out_ty.dtype
    return torch.quantize_per_tensor(
        input, qparams["scale"], qparams["zero_point"], dtype
    )


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_channel),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scales", "scales"),
        ("zero_points", "zero_points"),
        ("axis", "axis"),
        ("dtype", "dtype"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scales", "scale", move_to_qparams),
        ("zero_points", "zero_point", move_to_qparams),
        ("axis", "axis", move_to_qparams),
        ("dtype", "dtype", dont_move_to_qparams),
    ],
)
@register_acc_op
def quantize_per_channel(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    dtype = acc_out_ty.dtype
    return torch.quantize_per_channel(
        input,
        torch.tensor(qparams["scale"]),
        torch.tensor(qparams["zero_point"]),
        qparams["axis"],
        dtype,
    )  # type: ignore[call-overload]


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "dequantize"))
@register_acc_op_mapping(op_and_target=("call_function", torch.dequantize))
@register_acc_op
def dequantize(*, input):
    return torch.dequantize(input)


@register_acc_op_properties(
    AccOpProperty.pointwise, AccOpProperty.unary, AccOpProperty.quantized
)
@register_acc_op
def rescale_quantize_per_tensor(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    d = dequantize(input=input)
    return quantize_per_tensor(input=d, acc_out_ty=acc_out_ty)


@register_acc_op_properties(AccOpProperty.unary, AccOpProperty.quantized)
@register_acc_op
def rescale_quantize_per_channel(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    d = dequantize(input=input)
    return quantize_per_channel(input=d, acc_out_ty=acc_out_ty)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.sub))
@register_acc_op_mapping(op_and_target=("call_function", operator.sub))
@register_acc_op_mapping(op_and_target=("call_method", "sub"))
@register_acc_op
def sub(*, input, other):
    if not (isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor)):
        return operator.sub(input, other)
    else:
        return input - other


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.mul))
@register_acc_op_mapping(op_and_target=("call_function", operator.mul))
@register_acc_op_mapping(op_and_target=("call_method", "mul"))
@register_acc_op
def mul(*, input, other):
    return input * other


@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.aten.threshold_backward.default),
    arg_replacement_tuples=[
        ("grad", "grad"),
        ("self", "input"),
        ("threshold", "threshold"),
    ],
)
@register_acc_op
def threshold_backward(*, grad, input, threshold):
    return torch.ops.aten.threshold_backward.default(grad, input, threshold)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "div"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.div),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
def div_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        div_kwargs = dict(node.kwargs)
        if "rounding_mode" not in div_kwargs or div_kwargs["rounding_mode"] is None:
            div_node = node.graph.call_function(
                div, kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]}
            )
        elif div_kwargs["rounding_mode"] == "trunc":
            div_node = node.graph.call_function(
                trunc_div,
                kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]},
            )
        elif div_kwargs["rounding_mode"] == "floor":
            div_node = node.graph.call_function(
                floor_div,
                kwargs={"input": div_kwargs["input"], "other": div_kwargs["other"]},
            )
        else:
            raise RuntimeError(
                f"Unhandled div rounding mode {div_kwargs['rounding_mode']}"
            )
        div_node.meta = node.meta.copy()
        return div_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.truediv))
@register_acc_op
def div(*, input, other):
    return input / other


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.floordiv))
@register_acc_op
def floor_div(*, input, other):
    # This is temp fix because currently operator.floor_div for tensors would
    # traslate into torch.floor_divide which would throw an error. After it's
    # fixed we can stick to `input // other`.
    if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
        return torch.div(input, other, rounding_mode="floor")
    return input // other


# torch.floor_divide rounds result toward zero, rather than -Inf.
# https://github.com/pytorch/pytorch/issues/43874
@register_acc_op_mapping(op_and_target=("call_function", torch.floor_divide))
@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op
def trunc_div(*, input, other):
    return torch.div(input, other, rounding_mode="trunc")


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.pow))
@register_acc_op_mapping(op_and_target=("call_method", "pow"))
@register_acc_op
def pow(*, input, exponent):
    return torch.pow(input, exponent)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", nn.functional.relu))
@register_acc_op_mapping(
    op_and_target=("call_function", torch.relu),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "relu"),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op
def relu(*, input, inplace=False):
    return nn.functional.relu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.leaky_relu)
)
@register_acc_op
def leaky_relu(*, input, negative_slope=0.01, inplace=False):
    return nn.functional.leaky_relu(
        input=input, negative_slope=negative_slope, inplace=inplace
    )


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.prelu))
@register_acc_op
def prelu(*, input, weight):
    return nn.functional.prelu(
        input=input,
        weight=weight,
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.elu))
@register_acc_op
def elu(*, input, alpha=1.0, inplace=False):
    return nn.functional.elu(input=input, alpha=alpha, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.selu))
@register_acc_op
def selu(*, input, inplace=False):
    return nn.functional.selu(input=input, inplace=inplace)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.softsign))
@register_acc_op
def softsign(*, input):
    return nn.functional.softsign(input=input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.log1p),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def torch_log1p_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        add_kwargs = {"input": node.kwargs["input"], "other": 1.0}
        add_node = node.graph.call_function(add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()
        log_kwargs = {"input": add_node}
        log_node = node.graph.call_function(log, kwargs=log_kwargs)
        log_node.meta = node.meta.copy()
        return log_node


def reduce_op_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule, func
) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        if "dim" in kwargs and isinstance(kwargs["dim"], int):
            kwargs["dim"] = (kwargs["dim"],)
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def sum(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.sum(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "sum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.sum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def sum_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    return reduce_op_mapper(node, mod, sum)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def prod(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.prod(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.prod(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "prod"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.prod),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def prod_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    func = prod
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def mean(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.mean(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "mean"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.mean),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def mean_mapper(node, mod):
    return reduce_op_mapper(node, mod, mean)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "std"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("unbiased", "unbiased", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.std),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("unbiased", "unbiased", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def std_mapper(node, mod):
    """
    Formula of std: sqrt(sum(pow(X-mean(X))))/N)
    This op is mapped to a few existing ops
    """
    input_node = node.kwargs["input"]
    # unbiased = node.kwargs.get("unbiased")
    dim = node.kwargs.get("dim")
    keepdim = node.kwargs.get("keepdim")
    # assert unbiased is False or unbiased is None, "We currently do not support `std` with unbiased=True where n-1 is used"
    assert (
        dim is not None and keepdim is not None
    ), "We currently do not support `std` with dim=None and keepdim=None"

    with node.graph.inserting_before(node):
        # mean(X)
        mean_kwargs = {
            "input": input_node,
            "dim": dim,
            "keepdim": True,
        }
        mean_node = node.graph.call_function(mean, kwargs=mean_kwargs)
        mean_node.meta["type"] = torch.Tensor
        # X-mean(X)
        sub_kwargs = {
            "input": input_node,
            "other": mean_node,
        }
        sub_node = node.graph.call_function(sub, kwargs=sub_kwargs)
        sub_node.meta["type"] = torch.Tensor
        # pow(X-mean(X))
        pow_kwargs = {
            "input": sub_node,
            "exponent": 2.0,
        }
        pow_node = node.graph.call_function(pow, kwargs=pow_kwargs)
        pow_node.meta["type"] = torch.Tensor
        # mean(pow(X-mean(X)))
        post_mean_kwargs = {
            "input": pow_node,
            "dim": dim,
            "keepdim": keepdim,
        }
        post_mean_node = node.graph.call_function(mean, kwargs=post_mean_kwargs)
        post_mean_node.meta["type"] = torch.Tensor
        # sqrt( mean(pow(X-mean(X))) )
        sqrt_kwargs = {
            "input": post_mean_node,
        }
        sqrt_node = node.graph.call_function(sqrt, kwargs=sqrt_kwargs)
        sqrt_node.meta["type"] = torch.Tensor

        output_node = sqrt_node
        output_node.meta = node.meta.copy()
        return output_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "max"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.max),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "min"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.min),
    arg_replacement_tuples=[
        ("input", "input"),
        (("dim", "other"), "dim_or_other", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
def add_maximum_minimum_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule
) -> torch.fx.Node:
    # there are effectively three versions of torch.max / torch.min
    # full reduce: torch.max(input) -> Tensor
    # dimensional reduce: torch.max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
    # elementwise: torch.max(input, other, *, out=None) -> Tensor

    # the mapper function is remapping for both min and max situations
    # this helper function makes the choices available clearer and provides an easier way
    # to lookup the right function
    def target_map(op, target):
        if (op, target) in (("call_method", "max"), ("call_function", torch.max)):
            return {
                "full_reduce": max_full_reduce,
                "dim_reduce": max_dim_reduce,
                "elementwise": maximum,
            }
        elif (op, target) in (("call_method", "min"), ("call_function", torch.min)):
            return {
                "full_reduce": min_full_reduce,
                "dim_reduce": min_dim_reduce,
                "elementwise": minimum,
            }

    with node.graph.inserting_before(node):
        new_targets = target_map(node.op, node.target)
        max_kwargs = {}
        max_kwargs["input"] = node.kwargs["input"]
        if ("dim_or_other" not in node.kwargs) or (node.kwargs["dim_or_other"] is None):
            nt = new_targets["full_reduce"]
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        elif isinstance(node.kwargs["dim_or_other"], int):
            nt = new_targets["dim_reduce"]
            dim = node.kwargs["dim_or_other"]
            max_kwargs["dim"] = dim
            max_kwargs["keepdim"] = node.kwargs.get("keepdim", False)
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        else:
            other = node.kwargs["dim_or_other"]
            assert isinstance(other, torch.fx.Node)
            # Lowering path for when provided "other", where we do elem-wise max
            nt = new_targets["elementwise"]
            max_kwargs["other"] = other
            max_node = node.graph.call_function(nt, kwargs=max_kwargs)
        max_node.meta = node.meta.copy()
        return max_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def max_full_reduce(*, input):
    return torch.max(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def max_dim_reduce(*, input, dim=None, keepdim=False):
    return torch.max(input=input, dim=dim, keepdim=keepdim)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.maximum))
@register_acc_op_mapping(op_and_target=("call_method", "maximum"))
@register_acc_op
def maximum(*, input, other):
    return torch.maximum(input=input, other=other)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def min_full_reduce(*, input):
    return torch.min(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def min_dim_reduce(*, input, dim=None, keepdim=False):
    return torch.min(input, dim=dim, keepdim=keepdim)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.minimum))
@register_acc_op_mapping(op_and_target=("call_method", "minimum"))
@register_acc_op
def minimum(*, input, other):
    return torch.minimum(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.ne))
@register_acc_op_mapping(op_and_target=("call_function", torch.ne))
@register_acc_op_mapping(op_and_target=("call_method", "ne"))
@register_acc_op
def ne(*, input, other):
    return operator.ne(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.eq))
@register_acc_op_mapping(op_and_target=("call_function", torch.eq))
@register_acc_op_mapping(op_and_target=("call_method", "eq"))
@register_acc_op
def eq(*, input, other):
    return operator.eq(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.gt))
@register_acc_op_mapping(op_and_target=("call_function", torch.gt))
@register_acc_op_mapping(op_and_target=("call_method", "gt"))
@register_acc_op
def gt(*, input, other):
    return operator.gt(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.lt))
@register_acc_op_mapping(op_and_target=("call_function", torch.lt))
@register_acc_op_mapping(op_and_target=("call_method", "lt"))
@register_acc_op
def lt(*, input, other):
    return operator.lt(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.and_))
@register_acc_op_mapping(op_and_target=("call_method", "bitwise_and"))
@register_acc_op_mapping(op_and_target=("call_function", torch.bitwise_and))
def bitwise_and(*, input, other):
    return operator.and_(input, other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.logical_and))
@register_acc_op_mapping(op_and_target=("call_method", "logical_and"))
@register_acc_op
def logical_and(*, input, other):
    return torch.logical_and(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.or_))
@register_acc_op_mapping(op_and_target=("call_function", torch.logical_or))
@register_acc_op_mapping(op_and_target=("call_method", "logical_or"))
@register_acc_op
def logical_or(*, input, other):
    return torch.logical_or(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.logical_not))
@register_acc_op_mapping(op_and_target=("call_method", "logical_not"))
@register_acc_op
def logical_not(*, input):
    return torch.logical_not(input=input)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", operator.xor))
@register_acc_op_mapping(op_and_target=("call_function", torch.logical_xor))
@register_acc_op_mapping(op_and_target=("call_method", "logical_xor"))
@register_acc_op
def logical_xor(*, input, other):
    return torch.logical_xor(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.isinf))
@register_acc_op
def isinf(*, input):
    return torch.isinf(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def any(*, input, dim=None, keepdim=False):
    if dim is not None:
        return torch.any(input, dim, keepdim=keepdim)
    else:
        return torch.any(input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.any),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "any"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
    ],
)
def any_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(any, kwargs=node.kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(op_and_target=("call_function", torch.fmod))
@register_acc_op_mapping(op_and_target=("call_method", "fmod"))
@register_acc_op
def fmod(*, input, other):
    return torch.fmod(input=input, other=other)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sigmoid))
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.aten.sigmoid.default)
)
@register_acc_op_mapping(op_and_target=("call_method", "sigmoid"))
@register_acc_op
def sigmoid(*, input):
    return torch.sigmoid(input=input)


@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.aten.sigmoid_backward.default)
)
@register_acc_op_mapping(op_and_target=("call_method", "sigmoid_backward"))
@register_acc_op
# first argument's name needs to be input to use same_shape_and_dtype_as_input
def sigmoid_backward(*, input, dest):
    return torch.ops.aten.sigmoid_backward(grad_output=input, output=dest)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sinh))
@register_acc_op
def sinh(*, input):
    return torch.sinh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cosh))
@register_acc_op
def cosh(*, input):
    return torch.cosh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tanh))
@register_acc_op_mapping(op_and_target=("call_method", "tanh"))
@register_acc_op
def tanh(*, input):
    return torch.tanh(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.asin))
@register_acc_op
def asin(*, input):
    return torch.asin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.acos))
@register_acc_op
def acos(*, input):
    return torch.acos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.atan))
@register_acc_op
def atan(*, input):
    return torch.atan(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.exp))
@register_acc_op
def exp(*, input):
    return torch.exp(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.log))
@register_acc_op
def log(*, input):
    return torch.log(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.log_softmax),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def log_softmax(*, input, dim, dtype=None):
    """
    _stacklevel are ignored here.
    """
    return torch.nn.functional.log_softmax(input=input, dim=dim, dtype=dtype)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sqrt))
@register_acc_op_mapping(op_and_target=("call_method", "sqrt"))
@register_acc_op
def sqrt(*, input):
    return torch.sqrt(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.reciprocal))
@register_acc_op
def reciprocal(*, input):
    return torch.reciprocal(input=input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "rsqrt"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.rsqrt),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def rsqrt_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_kwargs = {
            "input": input_node,
        }
        sqrt_node = node.graph.call_function(sqrt, kwargs=new_kwargs)
        sqrt_node.meta["type"] = torch.Tensor
        new_kwargs = {
            "input": sqrt_node,
        }
        rec_node = node.graph.call_function(reciprocal, kwargs=new_kwargs)
        rec_node.meta["type"] = torch.Tensor
        output_node = rec_node
        output_node.meta = node.meta.copy()
        return output_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.abs))
@register_acc_op
def abs(*, input):
    return torch.abs(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", operator.neg))
@register_acc_op_mapping(op_and_target=("call_function", torch.neg))
@register_acc_op
def neg(*, input):
    if not isinstance(input, torch.Tensor):
        return operator.neg(input)
    else:
        return torch.neg(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.floor))
@register_acc_op
def floor(*, input):
    return torch.floor(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.ceil))
@register_acc_op
def ceil(*, input):
    return torch.ceil(input=input)


@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.pad))
@register_acc_op
def pad(*, input, pad: List[int], mode: str, value: float):
    return torch.nn.functional.pad(input=input, pad=pad, mode=mode, value=value)


@register_acc_op_mapping(op_and_target=("call_function", torch.conv1d))
@register_acc_op
def conv1d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.conv2d))
@register_acc_op
def conv2d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op
def quantized_conv2d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    padding_mode,
    acc_out_ty=None,
):
    qparams = acc_out_ty.qparams
    return torch.nn.quantized.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        scale=qparams["scale"],
        zero_point=qparams["zero_point"],
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.conv3d))
@register_acc_op
def conv3d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.conv_transpose2d)
)
@register_acc_op
def conv_transpose2d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.conv_transpose3d)
)
@register_acc_op
def conv_transpose3d(
    *,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return nn.functional.conv_transpose3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.batch_norm))
@register_acc_op
def batch_norm(
    *,
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    return nn.functional.batch_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.layer_norm))
@register_acc_op
def layer_norm(*, input, normalized_shape, weight, bias, eps):
    return nn.functional.layer_norm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


def argmin_max_mapper_impl(node: torch.fx.Node, largest: bool) -> torch.fx.Node:
    """
    Map torch.argmin or torch.argmax to acc_ops.flatten (depend on dim) + acc_ops.topk
    + acc_ops.getitem + acc_ops.squeeze (depends on keepdim).
    """
    input_node = node.kwargs["input"]
    dim = node.kwargs["dim"]
    keepdim = node.kwargs["keepdim"]

    if dim is None and keepdim:
        raise RuntimeError(
            "We currently don't support argmin/argmax with dim=None and keepdim=True"
        )

    with node.graph.inserting_before(node):
        if dim is None:
            flatten_kwargs = {
                "input": node.kwargs["input"],
                "start_dim": 0,
                "end_dim": -1,
            }
            flatten_node = node.graph.call_function(flatten, kwargs=flatten_kwargs)
            flatten_node.meta["type"] = torch.Tensor
            input_node = flatten_node
            dim = -1

        topk_kwargs = {
            "input": input_node,
            "k": 1,
            "dim": dim,
            "largest": largest,
            "sorted": False,
        }
        topk_node = node.graph.call_function(topk, kwargs=topk_kwargs)
        # It's actually more like NamedTuple but tuple here should be fine.
        topk_node.meta["type"] = tuple

        getitem_kwargs = {"input": topk_node, "idx": 1}
        getitem_node = node.graph.call_function(getitem, kwargs=getitem_kwargs)
        getitem_node.meta["type"] = torch.Tensor
        output_node = getitem_node

        if not keepdim:
            squeeze_kwargs = {"input": getitem_node, "dim": dim}
            output_node = node.graph.call_function(squeeze, kwargs=squeeze_kwargs)

        output_node.meta = node.meta.copy()
        return output_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.argmin),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("keepdim", "keepdim"),
    ],
)
def torch_argmin_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    """
    Map torch.argmin to acc_ops.flatten (depend on dim) + acc_ops.topk + acc_ops.getitem
    + acc_ops.squeeze (depends on keepdim).
    """
    return argmin_max_mapper_impl(node, largest=False)


@register_acc_op_mapping(op_and_target=("call_function", torch.linalg.norm))
@register_acc_op
def linalg_norm(*, input, ord, dim, keepdim):
    return torch.linalg.norm(input=input, ord=ord, dim=dim, keepdim=keepdim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.functional.norm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("p", "p"),
        ("dim", "dim"),
        ("keepdim", "keepdim"),
    ],
)
def norm_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    p = node.kwargs["p"]
    dim = node.kwargs["dim"]
    keepdim = node.kwargs["keepdim"]
    output_node = None
    with node.graph.inserting_before(node):
        if dim is None and p == 1:
            # linalg_norm takes the max along the sum along a dim
            # rather than the entire sum for p = 1
            abs_node = node.graph.call_function(abs, kwargs={"input": input_node})
            output_node = node.graph.call_function(
                sum,
                kwargs={"input": abs_node},
            )
        elif dim is None:
            raise RuntimeError("dim=None has not been implemented for p != 1")
        else:
            output_node = node.graph.call_function(
                linalg_norm,
                kwargs={"input": input_node, "ord": p, "dim": dim, "keepdim": keepdim},
            )

    return output_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split_with_sizes"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_sizes", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.split),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
def torch_split_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    If split_size_or_sections is sections, map the node to slice_tensors
    + tuple_construct. Otherwise, if split_size_or_sections is split_size,
    map the node to acc_ops.split.
    """
    split_size_or_sections = node.kwargs["split_size_or_sections"]
    with node.graph.inserting_before(node):
        if isinstance(split_size_or_sections, int):
            new_kwargs = {
                "input": node.kwargs["input"],
                "split_size": split_size_or_sections,
                "dim": node.kwargs["dim"],
            }
            new_node = node.graph.call_function(split, kwargs=new_kwargs)
            new_node.meta = node.meta.copy()
            return new_node

        assert isinstance(split_size_or_sections, Sequence)
        start = 0
        slice_nodes = []
        for i in split_size_or_sections:
            assert isinstance(i, int)
            new_kwargs = {
                "input": node.kwargs["input"],
                "dim": node.kwargs["dim"],
                "start": start,
                "stop": start + i,
                "step": 1,
            }
            new_node = node.graph.call_function(slice_tensor, kwargs=new_kwargs)
            new_node.meta["type"] = torch.Tensor
            slice_nodes.append(new_node)
            start += i

        new_node = node.graph.call_function(
            tuple_construct, kwargs={"tensors": tuple(slice_nodes)}
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def split(*, input, split_size, dim):
    return torch.split(input, split_size, dim)


@register_acc_op
def tuple_construct(*, tensors):
    return tuple(tensors)


@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.ops.quantized.batch_norm2d),
    arg_replacement_tuples=[
        ("input", "input"),
        ("weight", "weight"),
        ("bias", "bias"),
        ("running_mean", "running_mean"),
        ("running_var", "running_var"),
        ("eps", "eps"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
    ],
)
@register_acc_op
def quantized_batch_norm2d(
    *,
    input,
    running_mean,
    running_var,
    weight,
    bias,
    eps,
    acc_out_ty=None,
):
    qparams = acc_out_ty.qparams
    return torch.ops.quantized.batch_norm2d(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        qparams["scale"],
        qparams["zero_point"],
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.embedding_bag))
@register_acc_op
def embedding_bag(
    *,
    input,
    weight,
    offsets,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return nn.functional.embedding_bag(
        input=input,
        weight=weight,
        offsets=offsets,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        sparse=sparse,
        per_sample_weights=per_sample_weights,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )


@register_acc_op_mapping(
    op_and_target=(
        "call_function",
        torch.ops.quantized.embedding_bag_byte_rowwise_offsets,
    )
)
@register_acc_op
def embedding_bag_byte_rowwise_offsets(
    *,
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    pruned_weights,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
):
    return torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
        weight=weight,
        indices=indices,
        offsets=offsets,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        pruned_weights=pruned_weights,
        per_sample_weights=per_sample_weights,
        compressed_indices_mapping=compressed_indices_mapping,
        include_last_offset=include_last_offset,
    )


@register_acc_op_mapping(
    op_and_target=(
        "call_function",
        torch.ops.quantized.embedding_bag_4bit_rowwise_offsets,
    )
)
@register_acc_op
def embedding_bag_4bit_rowwise_offsets(
    *,
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    pruned_weights,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
):
    return torch.ops.quantized.embedding_bag_4bit_rowwise_offsets(
        weight=weight,
        indices=indices,
        offsets=offsets,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode,
        pruned_weights=pruned_weights,
        per_sample_weights=per_sample_weights,
        compressed_indices_mapping=compressed_indices_mapping,
        include_last_offset=include_last_offset,
    )


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.sin))
@register_acc_op_mapping(op_and_target=("call_method", "sin"))
@register_acc_op
def sin(*, input):
    return torch.sin(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cos))
@register_acc_op_mapping(op_and_target=("call_method", "cos"))
@register_acc_op
def cos(*, input):
    return torch.cos(input=input)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.tan))
@register_acc_op
def tan(*, input):
    return torch.tan(input=input)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.topk))
@register_acc_op
def topk(*, input, k, dim, largest, sorted):
    return torch.topk(input=input, k=k, dim=dim, largest=largest, sorted=sorted)


@register_acc_op_mapping(op_and_target=("call_function", operator.getitem))
@register_acc_op
def getitem(*, input, idx):
    return input[idx]


@register_acc_op
def nan_to_num(*, input, nan=None, posinf=None, neginf=None):
    return torch.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.nan_to_num),
    arg_replacement_tuples=[
        ("input", "input"),
        ("nan", "nan"),
        ("posinf", "posinf"),
        ("neginf", "neginf"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "nan_to_num"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("nan", "nan"),
        ("posinf", "posinf"),
        ("neginf", "neginf"),
    ],
)
def custom_nan_to_num_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    nan_val, posinf, neginf = (
        node.kwargs["nan"],
        node.kwargs["posinf"],
        node.kwargs["neginf"],
    )
    if nan_val is None:
        nan_val = 0
    if posinf is None:
        posinf = torch.finfo(torch.float16).max
    if neginf is None:
        neginf = torch.finfo(torch.float16).min
    kwargs = {
        "input": node.kwargs["input"],
        "nan": nan_val,
        "posinf": posinf,
        "neginf": neginf,
    }
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(nan_to_num, kwargs=kwargs)
    new_node.meta = node.meta.copy()
    return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_method", "expand"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "sizes"),
    ],
)
@register_acc_op
def expand(*, input, sizes):
    return input.expand(*sizes)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.masked_fill),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mask", "mask"),
        ("value", "value"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "masked_fill"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mask", "mask"),
        ("value", "value"),
    ],
)
@register_acc_op
def masked_fill(*, input, mask, value):
    return input.masked_fill(mask, value)


@register_acc_op_mapping(op_and_target=("call_function", torch.where))
@register_acc_op
def where(*, condition, x, y):
    return torch.where(condition, x, y)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def slice_tensor(*, input, dim, start, stop, step):
    slc = slice(start, stop, step)
    if dim >= 0:
        slices: List[slice] = [slice(None, None, None) for _ in range(dim)]
        slices.append(slc)
    else:
        slices = [Ellipsis, slc]  # type: ignore[list-item]
        slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

    return input[tuple(slices)]


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.narrow),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "narrow"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("start", "start"),
        ("length", "length"),
    ],
)
def custom_narrow_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    assert isinstance(node.kwargs["start"], int) and isinstance(
        node.kwargs["length"], int
    )
    kwargs = {
        "input": node.kwargs["input"],
        "dim": node.kwargs["dim"],
        "start": node.kwargs["start"],
        "stop": node.kwargs["start"] + node.kwargs["length"],
        "step": 1,
    }
    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(slice_tensor, kwargs=kwargs)
    new_node.meta = node.meta.copy()
    return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.reshape),
    arg_replacement_tuples=[
        ("input", "input"),
        ("shape", "shape"),
    ],
    kwargs_to_move_to_acc_out_ty=[("shape", "shape")],
)
@register_acc_op
def reshape(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    shape = acc_out_ty.shape
    if len(shape) == 1 and not isinstance(shape[0], int):
        return input.reshape(shape[0])
    return input.reshape(shape)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "reshape"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "view"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
def custom_tensor_reshape_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    For Tensor.reshape and Tensor.view nodes, args could be (input, 1, 2, 3) or (input,
    (1, 2, 3)).  Here we do some special handling with the `shape` arg in order to map
    it to acc_ops.reshape. It also handles the case when `shape` is a list instead of
    tuple.
    """
    input_node = node.kwargs["input"]
    shape = node.kwargs["shape"]

    assert isinstance(shape, Sequence)
    if isinstance(shape[0], (tuple, list)):  # type: ignore[index]
        shape = shape[0]  # type: ignore[index]

    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            reshape,
            kwargs={
                "input": input_node,
                "acc_out_ty": acc_utils.build_raw_tensor_meta(shape=shape),
            },
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "half"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def custom_half_mapper(node: torch.fx.Node, _: nn.Module):
    with node.graph.inserting_before(node):
        new_kwargs = {
            "input": node.kwargs["input"],
            "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=torch.float16),
        }
        new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "int"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def custom_int_mapper(node: torch.fx.Node, _: nn.Module):
    with node.graph.inserting_before(node):
        new_kwargs = {
            "input": node.kwargs["input"],
            "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=torch.int),
        }
        new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "float"),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def custom_float_mapper(node: torch.fx.Node, _: nn.Module):
    with node.graph.inserting_before(node):
        new_kwargs = {
            "input": node.kwargs["input"],
            "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=torch.float),
        }
        new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
        new_node.meta = node.meta
        return new_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op
def to_dtype(input, acc_out_ty=None, device=None):
    assert acc_out_ty is not None
    return input.to(dtype=acc_out_ty.dtype, device=device)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "to"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dtype", "dtype"),
        ("device", "device", this_arg_is_optional),
    ],
)
def custom_tensor_to_mapper(node: torch.fx.Node, _: nn.Module):
    dest = node.kwargs["dtype"]
    mem_format = node.kwargs.get("memory_format")
    dest_other = node.kwargs.get("device")
    assert dest is not None
    assert mem_format is None or mem_format == torch.preserve_format

    dest_dtype = dest_device = None
    if isinstance(dest, torch.fx.node.Node):
        meta_type = dest.meta["type"]
        # consider the device is gpu only, meta info is limited to give clear device type
        if dest.meta["type"] == torch.device:
            dest_device = dest
        elif dest.meta["type"] == torch.dtype:
            dest_dtype = dest
        elif dest.meta["type"] == torch.Tensor:
            input_obj = node.kwargs["input"]
            other_obj = dest
            with node.graph.inserting_before(node):
                dtype_node = node.graph.call_function(
                    dtype, kwargs={"input": other_obj}
                )
                dtype_node.meta["type"] = torch.dtype
                device_node = node.graph.call_function(
                    device, kwargs={"input": other_obj}
                )
                device_node.meta["type"] = torch.device
                new_kwargs = {
                    "input": input_obj,
                    "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=dtype_node),
                    "device": device_node,
                }
                new_node = node.graph.call_function(to_dtype, kwargs=new_kwargs)
                new_node.meta = node.meta
                return new_node
        else:
            raise RuntimeError(f"We currently do not support to({meta_type})")
    elif isinstance(dest, torch.device):
        # only device is set, dtype=None
        if dest_other is None:
            dest_device = dest
        # device and dtype are both set
        else:
            dest_dtype = dest_other
            dest_device = dest
    # only dtype is set
    else:
        dest_dtype = dest

    new_kwargs = {
        "input": node.kwargs["input"],
        "acc_out_ty": acc_utils.build_raw_tensor_meta(dtype=dest_dtype),
        "device": dest_device,
    }

    with node.graph.inserting_before(node):
        new_node = node.graph.create_node(
            "call_function", to_dtype, kwargs=new_kwargs, name=node.name
        )
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.add),
    # Note that we may have aliases for inputs here due to issues with deterministically
    # knowing the correct target that will be resolved by pytorch.
    arg_replacement_tuples=[
        (("input", "a"), "input"),
        (("other", "b"), "other"),
        ("alpha", "alpha", this_arg_is_optional),
    ],
)
def custom_torch_add_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    Add custom mapping for torch.add because it has an `alpha` parameter which scales
    the `other` input, and we want to make that mul a separate node.
    """
    with node.graph.inserting_before(node):
        # If alpha is in kwargs check if we need to add a mul, and use correct kwargs.
        if "alpha" in node.kwargs:
            # Add mul node only if it has a numerical impact, i.e. alpha != 1.0.
            if node.kwargs["alpha"] != 1.0:
                other_node = node.graph.create_node(
                    "call_function",
                    mul,
                    kwargs={
                        "input": node.kwargs["other"],
                        "other": node.kwargs["alpha"],
                    },
                    name=node.name + "_mul_alpha",
                )
                other_node.meta = node.meta
            else:
                other_node = node.kwargs["other"]
            add_kwargs = {"input": node.kwargs["input"], "other": other_node}
        else:
            add_kwargs = node.kwargs

        new_node = node.graph.create_node(
            "call_function", add, kwargs=add_kwargs, name=node.name
        )
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.quantized.Linear),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_linear_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantized_linear module to acc_op.linear. We unpack weight and bias
    in this mapper and pass them directly to linear node.
    """
    assert isinstance(node.target, str)
    linear_module = dict(mod.named_modules())[node.target]
    prefix = node.target.replace(".", "_")
    weight_name = f"{prefix}_weight"
    bias_name = f"{prefix}_bias"

    # Store weight and bias in the main module
    mod.register_buffer(weight_name, linear_module.weight())
    if linear_module.bias() is not None:
        mod.register_buffer(bias_name, linear_module.bias())

    with node.graph.inserting_before(node):
        # Insert get_attr nodes for weight and bias
        get_weight = node.graph.get_attr(weight_name)
        get_weight.meta["tensor_meta"] = _extract_tensor_metadata(
            linear_module.weight()
        )

        get_bias = None
        if linear_module.bias() is not None:
            get_bias = node.graph.get_attr(bias_name)
            get_bias.meta["tensor_meta"] = _extract_tensor_metadata(
                linear_module.bias()
            )

        qparams = {"scale": linear_module.scale, "zero_point": linear_module.zero_point}
        # Create kwargs for acc_op.quantized_linear
        kwargs = {
            "input": node.kwargs["input"],
            "weight": get_weight,
            "bias": get_bias,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }

        new_node = node.graph.call_function(quantized_linear, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.quantized.Conv2d),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_conv2d_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantzed Conv2d module to acc_op.conv. We unpack all the parameters
    in this mapper and pass them directly to conv2d node.
    """
    assert isinstance(node.target, str)
    conv_module = dict(mod.named_modules())[node.target]
    prefix = node.target.replace(".", "_")
    weight_name = f"{prefix}_weight"
    bias_name = f"{prefix}_bias"

    # Store weight and bias in the main module
    mod.register_buffer(weight_name, conv_module.weight())
    if conv_module.bias() is not None:
        mod.register_buffer(bias_name, conv_module.bias())

    with node.graph.inserting_before(node):
        # Insert get_attr nodes for weight and bias
        get_weight = node.graph.get_attr(weight_name)
        get_weight.meta["tensor_meta"] = _extract_tensor_metadata(conv_module.weight())

        get_bias = None
        if conv_module.bias() is not None:
            get_bias = node.graph.get_attr(bias_name)
            get_bias.meta["tensor_meta"] = _extract_tensor_metadata(conv_module.bias())

        qparams = {"scale": conv_module.scale, "zero_point": conv_module.zero_point}

        # Create kwargs for acc_op.conv
        kwargs = {
            "input": node.kwargs["input"],
            "weight": get_weight,
            "bias": get_bias,
            "stride": conv_module.stride,
            "padding": conv_module.padding,
            "dilation": conv_module.dilation,
            "groups": conv_module.groups,
            "padding_mode": conv_module.padding_mode,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }

        new_node = node.graph.call_function(quantized_conv2d, kwargs=kwargs)
        new_node.meta = node.meta
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.ops.quantized.add_relu),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
    ],
)
def add_relu_unfuse_mapper(
    node: torch.fx.Node, mod: torch.fx.GraphModule
) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        qparams = {
            "scale": node.kwargs["scale"],
            "zero_point": node.kwargs["zero_point"],
        }
        add_kwargs = {
            "input": node.kwargs["input"],
            "other": node.kwargs["other"],
            "acc_out_ty": acc_utils.build_raw_tensor_meta(qparams=qparams),
        }
        add_node = node.graph.call_function(quantized_add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()

        relu_node = node.graph.call_function(
            relu, kwargs={"input": add_node, "inplace": False}
        )
        relu_node.meta = node.meta
        return relu_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", nn.intrinsic.quantized.ConvReLU2d),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def packed_quantized_convrelu2d_mapper(
    node: torch.fx.Node, mod: nn.Module
) -> torch.fx.Node:
    """
    Mapping from quantized ConvReLU2d module to acc_op.relu. We use packed_quantized_conv2d_mapper to unpack all the parameters
    in this mapper and pass the returned conv2d node directly to relu node.
    """

    with node.graph.inserting_before(node):
        # conv2d op
        conv2d_node = packed_quantized_conv2d_mapper(node, mod)

        # relu op
        relu_node = node.graph.call_function(
            relu, kwargs={"input": conv2d_node, "inplace": False}
        )
        relu_node.meta = node.meta
        return relu_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.nn.functional.gelu))
@register_acc_op_mapping(op_and_target=("call_method", "gelu"))
@register_custom_acc_mapper_fn(
    op_and_target=("call_module", torch.nn.GELU),
    arg_replacement_tuples=[
        ("input", "input"),
        ("approximate", "approximate"),
    ],
)
@register_acc_op
def gelu(*, input, approximate="none"):
    return torch.nn.functional.gelu(input=input, approximate=approximate)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.cumsum))
@register_acc_op_mapping(
    op_and_target=("call_method", "cumsum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def cumsum(*, input, dim, dtype=None):
    return torch.cumsum(input=input, dim=dim, dtype=dtype)


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_function", torch.chunk))
@register_acc_op_mapping(op_and_target=("call_method", "chunk"))
@register_acc_op
def chunk(*, input, chunks, dim=0):
    return torch.chunk(input=input, chunks=chunks, dim=dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.gather),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("index", "index"),
        ("sparse_grad", "sparse_grad", this_arg_is_optional),
    ],
)
@register_acc_op
def gather(*, input, dim, index, sparse_grad=False):
    return torch.gather(input=input, dim=dim, index=index, sparse_grad=sparse_grad)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.index_select),
)
@register_acc_op
def index_select(*, input, dim, index):
    return torch.index_select(input, dim, index)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "expand_as"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
    ],
)
def expand_as_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Maps expand_as(other) to expand(other.size())
    """
    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(
            size, kwargs={"input": node.kwargs["other"]}
        )
        size_node.meta["type"] = torch.Size

        expand_node = node.graph.call_function(
            expand, kwargs={"input": node.kwargs["input"], "sizes": size_node}
        )
        expand_node.meta = node.meta.copy()
        return expand_node


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.grid_sample),
    arg_replacement_tuples=[
        ("input", "input"),
        ("grid", "grid"),
        ("mode", "mode", this_arg_is_optional),
        ("padding_mode", "padding_mode", this_arg_is_optional),
        ("align_corners", "align_corners", this_arg_is_optional),
    ],
)
@register_acc_op
def grid_sample(
    *,
    input,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=None,
):
    return torch.nn.functional.grid_sample(
        input=input,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.interpolate),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size", this_arg_is_optional),
        ("scale_factor", "scale_factor", this_arg_is_optional),
        ("mode", "mode", this_arg_is_optional),
        ("align_corners", "align_corners", this_arg_is_optional),
    ],
)
@register_acc_op
def interpolate(
    *,
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
):
    return torch.nn.functional.interpolate(
        input=input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


@register_acc_op_mapping(
    op_and_target=("call_function", torch.tensor_split),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"), "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "tensor_split"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"), "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def tensor_split(*, input, indices_or_sections, dim=0):
    # Need to de-coalesce the indices_or_sections because tensor_split accepts
    # one of three kwarg signatures:
    #  * (Tensor input, Tensor tensor_indices_or_sections, int dim)
    #  * (Tensor input, int sections, int dim)
    #  * (Tensor input, tuple of ints indices, int dim)
    if isinstance(indices_or_sections, torch.Tensor):
        indices_or_sections = indices_or_sections.tolist()
    if isinstance(indices_or_sections, int):
        return torch.tensor_split(input, sections=indices_or_sections, dim=dim)
    elif isinstance(indices_or_sections, Iterable):
        return torch.tensor_split(input, indices=tuple(indices_or_sections), dim=dim)
    else:
        raise RuntimeError(
            f"Expected int, Iterable or Tensor for "
            f"indices_or_sections arg, got: {type(indices_or_sections)}"
        )


@register_acc_op_mapping(
    op_and_target=("call_method", "new_empty"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("dtype", "dtype", this_arg_is_optional),
        ("device", "device", this_arg_is_optional),
        ("requires_grad", "requires_grad", this_arg_is_optional),
    ],
)
@register_acc_op
def new_empty(*, input, size, dtype=None, device=None, requires_grad=False):
    assert requires_grad is False, f"requires_grad != False, it is {requires_grad}"
    return input.new_empty(size, dtype=dtype, device=device)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.einsum),
    arg_replacement_tuples=[
        ("equation", "equation"),
        ("*", "operands"),
    ],
)
@register_acc_op
def einsum(*, equation, operands):
    return torch.einsum(equation, *operands)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.as_strided),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("stride", "stride"),
        ("storage_offset", "storage_offset", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "as_strided"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("stride", "stride"),
        ("storage_offset", "storage_offset", this_arg_is_optional),
    ],
)
@register_acc_op
def as_strided(*, input, size, stride, storage_offset=0):
    return torch.as_strided(
        input=input, size=size, stride=stride, storage_offset=storage_offset
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.var))
@register_acc_op_mapping(
    op_and_target=("call_method", "var"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("unbiased", "unbiased"),
        ("keepdim", "keepdim"),
    ],
)
@register_acc_op
def var(*, input, dim, unbiased, keepdim=False):
    return torch.var(input=input, dim=dim, unbiased=unbiased, keepdim=keepdim)


@register_acc_op
def xl_weight(weight_id: str, metadata: TensorMetadata, proxy_shape, dtype):
    """
    This op stores metadata and weight_id and otherwise returns a zeros tensor
    with shape `proxy_shape` and dtype `dtype`.

    Note: when Nodes with this op are run through ShapeProp, its metadata will
    be the same as computed and set as of that of `proxy`, however when running
    acc_shape_inference, it will return `metadata`.

    Args:
        weight_id: string identifier for the XL weight
        metadata: metadata of the XL weight
        proxy_shape: shape of substitute tensor
        dtype: dtype of substitute tensor
    """
    return torch.zeros(proxy_shape, dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.nn.functional.softplus),
    arg_replacement_tuples=[
        ("input", "input"),
        ("beta", "beta", this_arg_is_optional),
        ("threshold", "threshold", this_arg_is_optional),
    ],
)
def softplus_mapper(node: torch.fx.Node, _: torch.nn.Module) -> torch.fx.Node:
    """
    Maps torch.nn.functional.softplus to acc_ops.where, acc_ops.relu, acc_ops.exp, acc_ops.mul, acc_ops.add and acc_ops.div

    softplus(input, beta, threshold) = where(beta * input > threshold, relu(input), div(log(1 + exp(beta * input))), beta))

    torch.where(
        softplus_module.beta * sample_inputs[0] > softplus_module.threshold,
        sample_inputs[0].relu(),
        torch.div((1 + (softplus_module.beta * sample_inputs[0]).exp()).log(), softplus_module.beta),
    )

    """

    input_node = node.kwargs["input"]
    beta_node = node.kwargs["beta"]
    threshold_node = node.kwargs["threshold"]

    with node.graph.inserting_after(node):
        cond_mul_node = node.graph.call_function(
            mul,
            kwargs={
                "input": input_node,
                "other": beta_node,
            },
        )
        cond_mul_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(cond_mul_node):
        gt_node = node.graph.call_function(
            gt,
            kwargs={
                "input": cond_mul_node,
                "other": threshold_node,
            },
        )
        gt_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(gt_node):
        relu_node = node.graph.call_function(relu, kwargs={"input": input_node})
        relu_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(relu_node):
        mul_node = node.graph.call_function(
            mul,
            kwargs={
                "input": input_node,
                "other": beta_node,
            },
        )
        mul_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(mul_node):
        exp_node = node.graph.call_function(exp, kwargs={"input": mul_node})
        exp_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(exp_node):
        add_node = node.graph.call_function(
            add,
            kwargs={
                "input": exp_node,
                "other": 1,
            },
        )
        add_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(add_node):
        log_node = node.graph.call_function(log, kwargs={"input": add_node})
        log_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(log_node):
        div_node = node.graph.call_function(
            div,
            kwargs={
                "input": log_node,
                "other": beta_node,
            },
        )
        div_node.meta = input_node.meta.copy()

    with node.graph.inserting_after(div_node):
        where_node = node.graph.call_function(
            where,
            kwargs={
                "condition": gt_node,
                "x": relu_node,
                "y": div_node,
            },
        )
        where_node.meta = div_node.meta.copy()

        return where_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.baddbmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("batch1", "batch1"),
        ("batch2", "batch2"),
        ("beta", "beta", this_arg_is_optional),
        ("alpha", "alpha", this_arg_is_optional),
    ],
)
def baddbmm_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from torch.baddbmm to acc_ops.mm -> acc_ops.add, if alpha or beta is not 1
    then we also insert acc_ops.mul to the right place.
    """
    with node.graph.inserting_before(node):
        mm_kwargs = {"input": node.kwargs["batch1"], "other": node.kwargs["batch2"]}
        mm_node = node.graph.create_node(
            "call_function", matmul, kwargs=mm_kwargs, name=f"{node.name}_matmul"
        )
        mm_node.meta = node.meta.copy()

        if node.kwargs["alpha"] != 1:
            mul_kwargs = {"input": mm_node, "other": node.kwargs["alpha"]}
            mm_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{mm_node.name}_mul"
            )
        mm_node.meta = node.meta.copy()

        input_node = node.kwargs["input"]
        if node.kwargs["beta"] != 1:
            mul_kwargs = {"input": input_node, "other": node.kwargs["beta"]}
            new_input_node = node.graph.create_node(
                "call_function", mul, kwargs=mul_kwargs, name=f"{node.name}_input_mul"
            )
            assert isinstance(input_node, torch.fx.Node)
            new_input_node.meta = input_node.meta.copy()
            input_node = new_input_node

        add_kwargs = {"input": input_node, "other": mm_node}
        add_node = node.graph.create_node(
            "call_function", add, kwargs=add_kwargs, name=f"{node.name}_add"
        )
        add_node.meta = node.meta.copy()
        return add_node


@register_acc_op_mapping(op_and_target=("call_function", torch.clone))
@register_acc_op_mapping(op_and_target=("call_method", "clone"))
@register_acc_op
def clone(*, input):
    return torch.clone(input)


@register_acc_op_mapping(op_and_target=("call_function", torch.unbind))
@register_acc_op
def unbind(*, input, dim=0):
    return torch.unbind(input, dim=dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.group_norm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("num_groups", "num_groups"),
        ("weight", "weight"),
        ("bias", "bias"),
        ("eps", "eps"),
    ],
)
@register_acc_op
def group_norm(*, input, num_groups, weight=None, bias=None, eps=1e-05):
    return torch.nn.functional.group_norm(
        input, num_groups, weight=weight, bias=bias, eps=eps
    )


@register_acc_op_mapping(op_and_target=("call_method", "long"))
@register_acc_op
def long(*, input):
    return input.long()


@register_acc_op_mapping(
    op_and_target=("call_method", "new_full"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("fill_value", "fill_value"),
        ("dtype", "dtype", this_arg_is_optional),
        ("device", "device", this_arg_is_optional),
        ("requires_grad", "requires_grad", this_arg_is_optional),
    ],
)
@register_acc_op
def new_full(*, input, size, fill_value, dtype=None, device=None, requires_grad=False):
    return input.new_full(size, fill_value=fill_value, dtype=dtype, device=device)


@register_acc_op_mapping(op_and_target=("call_function", torch.full_like))
@register_acc_op
def full_like(*, input, fill_value, dtype=None, device=None):
    return torch.full_like(
        input=input, fill_value=fill_value, dtype=dtype, device=device
    )


@register_acc_op_mapping(
    op_and_target=("call_method", "new_ones"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("dtype", "dtype", this_arg_is_optional),
        ("device", "device", this_arg_is_optional),
        ("requires_grad", "requires_grad", this_arg_is_optional),
    ],
)
@register_acc_op
def new_ones(*, input, size, dtype=None, device=None, requires_grad=False):
    assert requires_grad is False, f"requires_grad != False, it is {requires_grad}"
    return input.new_ones(size, dtype=dtype, device=device)


@register_acc_op_mapping(op_and_target=("call_function", torch.ones_like))
@register_acc_op
def ones_like(*, input, dtype=None, device=None):
    return torch.ones_like(input=input, dtype=dtype, device=device)


@register_acc_op_mapping(
    op_and_target=("call_method", "new_zeros"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
        ("dtype", "dtype", this_arg_is_optional),
        ("device", "device", this_arg_is_optional),
        ("requires_grad", "requires_grad", this_arg_is_optional),
    ],
)
@register_acc_op
def new_zeros(*, input, size, dtype=None, device=None, requires_grad=False):
    return input.new_zeros(size, dtype=dtype, device=device)


@register_acc_op_mapping(op_and_target=("call_function", torch.zeros_like))
@register_acc_op
def zeros_like(*, input, dtype=None, device=None):
    return torch.zeros_like(input=input, dtype=dtype, device=device)


@register_acc_op_mapping(
    op_and_target=("call_method", "index_add_"),
)
@register_acc_op_mapping(op_and_target=("call_function", torch.index_add))
@register_acc_op
def index_add(*, input, dim, index, source, alpha=1):
    return torch.index_add(input, dim, index, source, alpha=alpha)


@register_acc_op_mapping(op_and_target=("call_function", torch.masked_select))
@register_acc_op
def masked_select(*, input, mask):
    return torch.masked_select(input=input, mask=mask)


###############################################################################

# Set ops as side-effectul, this prevents them from being optimized away or
# being folded into constants.
torch.fx.node._side_effectful_functions.add(xl_weight)
