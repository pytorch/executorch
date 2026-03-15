# mypy: disallow-untyped-decorators=False

import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Argument, Node, Target
from torch_tensorrt import ENABLED_FEATURES
from torch_tensorrt._features import needs_not_tensorrt_rtx
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
    has_static_shapes_in_args,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    args_bounds_check,
    enforce_tensor_types,
    get_positive_dim,
    is_only_operator_on_placeholder,
)
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

_LOGGER: logging.Logger = logging.getLogger(__name__)


def get_ir(target: Target) -> SourceIR:
    target_module = getattr(target, "__module__", "None")
    if any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.aten", "torch._ops.aten")
    ):
        return SourceIR.ATEN
    elif any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.prims", "torch._ops.prims")
    ):
        return SourceIR.PRIM
    elif target_module.startswith("torch.nn"):
        return SourceIR.NN

    return SourceIR.UNKNOWN


def one_user_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    # Validate only one user, which is a getitem node that accesses the first element in the list
    return (
        len(node.users) == 1
        and list(node.users)[0].target == operator.getitem
        and list(node.users)[0].args[1] == 0
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_batch_norm.default,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.batch_norm.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(torch.ops.aten.batch_norm, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_batch_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        running_mean=args[3],
        running_var=args[4],
        training=args[5],
        momentum=args[6],
        eps=args[7],
        cudnn_enabled=args_bounds_check(args, 8, True),
        return_mean_rstd=(target == torch.ops.aten.native_batch_norm.default),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_batch_norm_legit_no_training(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        running_mean=args[3],
        running_var=args[4],
        training=False,
        momentum=args[5],
        eps=args[6],
        cudnn_enabled=False,
        return_mean_rstd=(
            target == torch.ops.aten._native_batch_norm_legit_no_training.default
        ),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._native_batch_norm_legit.no_stats,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_batch_norm_legit_no_stats(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        training=False,
        momentum=args[4],
        eps=args[5],
        return_mean_rstd=True,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_layer_norm.default,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_native_layer_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.native_layer_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        normalized_shape=args[1],
        weight=args_bounds_check(args, 2),
        bias=args_bounds_check(args, 3),
        eps=args[4],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_group_norm.default,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_native_group_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.native_group_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args_bounds_check(args, 1),
        bias=args_bounds_check(args, 2),
        N=args[3],
        C=args[4],
        HxW=args[5],
        group=args[6],
        eps=args[7],
    )


def parse_cat_args(
    args: Tuple[Argument, ...], kwargs: Dict[str, Any]
) -> Tuple[List[Any], int]:
    """
    Process inputs for torch.ops.aten.cat.default.

    Handles these valid patterns:
      1. args = ((t1, t2, ...), dim)
      2. args = ((t1, t2, ...),), kwargs = {dim: X} with optional dim in kwargs

    Returns:
        (input_tensors, dim)
        input_tensors: tuple of tensor arguments
        dim: integer concatenation dimension (default 0)
    """

    if len(args) > 1 and isinstance(args[0], (list, tuple)):
        input_tensors = list(args[0])
        dim = args_bounds_check(args, 1, 0)

    else:
        # If single arg is itself a tuple/list, unwrap it
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            input_tensors = list(args[0])
        else:
            input_tensors = list(args)

        dim = kwargs.get("dim", 0)

    return input_tensors, dim


@dynamo_tensorrt_converter(
    torch.ops.aten.cat.default,
    supports_dynamic_shapes=True,
)
def aten_ops_cat(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    inputs, dim = parse_cat_args(args, kwargs)
    return impl.cat.cat(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=inputs,
        dim=dim,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.embedding.default,
    supports_dynamic_shapes=True,
)
def aten_ops_embedding(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.embedding.embedding(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[1],
        weight=args[0],
    )


def embedding_bag_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    # Embedding bag op is not refitable
    if settings and not settings.immutable_weights:
        return False

    if not one_user_validator(node):
        return False
    meta = node.args[1].meta
    indices = meta.get("tensor_meta")
    if indices is None:
        indices = meta.get("val")
    if indices is None:
        return False
    return len(indices.shape) == 1  # currently only support 1D indices


@dynamo_tensorrt_converter(
    torch.ops.aten.embedding_bag.default,
    capability_validator=embedding_bag_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten._embedding_bag.default,
    capability_validator=embedding_bag_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
    }
)
def aten_ops_embedding_bag(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.embedding.embedding_bag(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        weight=args[0],
        indices=args[1],
        offsets=args[2],
        mode=args_bounds_check(args, 4, 0),
        per_sample_weights=args_bounds_check(args, 6, None),
        include_last_offset=args_bounds_check(args, 7, False),
    )


@dynamo_tensorrt_converter(operator.mod, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Scalar, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Tensor, supports_dynamic_shapes=True)
def aten_ops_fmod(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.grid_sampler, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.grid_sampler_2d, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(
    torch.ops.aten.grid_sampler.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.grid_sampler_2d.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
    }
)
def aten_ops_grid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.grid.grid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        grid=args[1],
        interpolation_mode=args[2],
        padding_mode=args[3],
        align_corners=args[4],
    )


@dynamo_tensorrt_converter(torch.ops.aten.relu.default, supports_dynamic_shapes=True)
def aten_ops_relu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.relu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sigmoid.default, supports_dynamic_shapes=True)
def aten_ops_sigmoid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.sigmoid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
@dynamo_tensorrt_converter(torch.ops.aten.sym_size.int, supports_dynamic_shapes=True)
def aten_ops_symsize_int(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shape.shape(ctx, target, SourceIR.ATEN, name, args[0], args[1])


def index_dtype_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    index = node.args[1]
    for ind in index:
        if ind is not None:
            val = ind.meta.get("val")
            if val is not None and val.dtype not in (
                torch.int32,
                torch.int64,
                torch.bool,
            ):
                return False
    return True


def index_nonbool_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    # for thor and tensorrt_rtx, we don't support boolean indices, due to nonzero op not supported
    if ENABLED_FEATURES.tensorrt_rtx:
        index = node.args[1]
        for ind in index:
            if ind is not None:
                val = ind.meta.get("val")
                if val is not None and val.dtype == torch.bool:
                    return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.index.Tensor,
    capability_validator=lambda node, settings: index_dtype_validator(node, settings)
    and index_nonbool_validator(node, settings),
    supports_dynamic_shapes=True,
    requires_output_allocator=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_index(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.index(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tanh.default, supports_dynamic_shapes=True)
def aten_ops_tanh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.tanh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.leaky_relu.default, supports_dynamic_shapes=True
)
def aten_ops_leaky_relu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.leaky_relu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, 0.01),
    )


@dynamo_tensorrt_converter(torch.ops.aten.elu.default, supports_dynamic_shapes=True)
def aten_ops_elu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.elu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1, 1.0),
        beta=args_bounds_check(args, 2, None),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.softplus.default, supports_dynamic_shapes=True
)
def aten_ops_softplus(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.softplus(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        beta=args_bounds_check(args, 1, 1),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.hardsigmoid.default, supports_dynamic_shapes=True
)
def aten_ops_hard_sigmoid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.hard_sigmoid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1, 1 / 6),
        beta=args_bounds_check(args, 2, 1 / 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.gelu.default, supports_dynamic_shapes=True)
def aten_ops_gelu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.gelu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("approximate", "none"),
    )


@dynamo_tensorrt_converter(torch.ops.aten.matmul, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.matmul.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.dot.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.mm.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.mv.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.bmm.default, supports_dynamic_shapes=True)
def aten_ops_matmul(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.matmul.matrix_multiply(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default, supports_dynamic_shapes=True)
def aten_ops_rsqrt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.rsqrt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.neg.default, supports_dynamic_shapes=True)
def aten_ops_neg(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.neg(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


try:
    import modelopt.torch.quantization as mtq  # noqa: F401

    assert torch.ops.tensorrt.quantize_op.default
except Exception as e:
    _LOGGER.warning(
        "Unable to import quantization op. Please install modelopt library (https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#installation) to add support for compiling quantized models"
    )
else:

    @dynamo_tensorrt_converter(
        torch.ops.tensorrt.quantize_op.default, supports_dynamic_shapes=True
    )
    def aten_ops_quantize_op(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[TRTTensor, Sequence[TRTTensor]]:
        return impl.quantize.quantize(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            args[0],
            args[1],
            args[2],
            args[3],
        )


if is_tensorrt_version_supported("10.8.0"):
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401

        assert torch.ops.tensorrt.dynamic_block_quantize_op.default
    except Exception as e:
        _LOGGER.warning(
            "Unable to import quantize op. Please install modelopt library (https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#installation) to add support for compiling quantized models"
        )
    else:

        @dynamo_tensorrt_converter(
            torch.ops.tensorrt.dynamic_block_quantize_op.default,
            supports_dynamic_shapes=True,
        )
        def aten_ops_dynamic_block_quantize_op(
            ctx: ConversionContext,
            target: Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
            name: str,
        ) -> Union[TRTTensor, Sequence[TRTTensor]]:
            return impl.dynamic_block_quantize.quantize(
                ctx,
                target,
                SourceIR.ATEN,
                name,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
            )


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims, supports_dynamic_shapes=True)
def aten_ops_squeeze(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.erf.default, supports_dynamic_shapes=True)
def aten_ops_erf(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.erf(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.unsqueeze.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_unsqueeze(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unsqueeze.unsqueeze(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(
    torch.ops.aten._softmax.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_softmax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.softmax(
        ctx, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.split.Tensor,
    capability_validator=has_static_shapes_in_args([1]),
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split.sizes,
    capability_validator=has_static_shapes_in_args([1]),
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split_with_sizes.default,
    capability_validator=has_static_shapes_in_args([1]),
    supports_dynamic_shapes=True,
)
def aten_ops_split(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.split.split(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        split_size_or_sections=args[1],
        dim=args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(torch.ops.aten.where.self, supports_dynamic_shapes=True)
def aten_ops_where(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.condition.where(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[1],
        args[2],
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.clamp.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.clip.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.clip.Tensor, supports_dynamic_shapes=True)
def aten_ops_clamp(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.clamp(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input_val=args[0],
        min_val=args_bounds_check(args, 1),
        max_val=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.gather.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        2: (TRTTensor,),
    }
)
def aten_ops_gather(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.gather(
        ctx, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(torch.ops.aten.scatter.src)
@dynamo_tensorrt_converter(torch.ops.aten.scatter.value)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        2: (TRTTensor,),
    }
)
def aten_ops_scatter(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.scatter(
        ctx, target, SourceIR.ATEN, name, args[0], args[1], args[2], args[3]
    )


@dynamo_tensorrt_converter(torch.ops.aten.select.int, supports_dynamic_shapes=True)
def aten_ops_select(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.select(
        ctx, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.index_put.default,
    capability_validator=lambda node, settings: index_dtype_validator(node, settings)
    and index_nonbool_validator(node, settings),
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        2: (TRTTensor,),
    }
)
def aten_ops_index_put(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.index_put_converter(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args_bounds_check(args, 3, False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_slice(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.slice_op(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=0),
        args_bounds_check(args, 2, replacement=None),
        args_bounds_check(args, 3, replacement=None),
        args_bounds_check(args, 4, replacement=1),
    )


def refit_validator(node: Node, settings: Optional[CompilationSettings] = None) -> bool:
    # cumsum op is not refitable
    if settings and not settings.immutable_weights:
        return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.cumsum.default,
    capability_validator=refit_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_cumsum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.cumsum(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tile.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_tile(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.tile(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def zero_output_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    if 0 in node.args[1]:
        _LOGGER.debug(
            f"We do not support output tensor {node.args[1]} tensors with zero-sized dimensions for this operation."
        )
        return False
    else:
        return True


@dynamo_tensorrt_converter(
    torch.ops.aten.as_strided.default,
    capability_validator=zero_output_validator,
)
def aten_ops_as_strided(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.as_strided(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        size=args[1],
        stride=args[2],
        storage_offset=args_bounds_check(args, 3, None),
    )


@dynamo_tensorrt_converter(torch.ops.aten.permute.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_permute(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.permutation.permute(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def to_copy_dtype_validator(
    placeholder_only: bool, settings: Optional[CompilationSettings] = None
) -> Callable[[Node, CompilationSettings], bool]:
    """Return validator for to_copy node with placeholder restrictions"""

    def validate_dtype(to_copy_node: Node) -> bool:
        """Returns true if the to_copy node can be converted to TRT

        Based on data type being casted to
        """
        allowed_casts = {
            torch.float,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.int8,
            torch.float16,
            torch.bfloat16,
        }

        # Validate input node has convertible kwargs
        if "dtype" in to_copy_node.kwargs:
            if to_copy_node.kwargs["dtype"] in allowed_casts:
                return True
            else:
                _LOGGER.debug(
                    f"_to_copy converter rejected node {to_copy_node} with dtype {to_copy_node.kwargs['dtype']}"
                )
                return False
        else:
            _LOGGER.debug(
                f"_to_copy converter rejected node {to_copy_node} with kwargs {to_copy_node.kwargs}"
            )
            return False

    def validator(
        to_copy_node: Node, settings: Optional[CompilationSettings] = None
    ) -> bool:
        """Returns true if the to_copy node can be converted to TRT
        and the placeholder restriction is satisfied
        """
        # The placeholder restriction is satsfied if placeholder_only is the same
        # truth value as is_only_operator_on_placeholder(to_copy_node)
        return validate_dtype(to_copy_node) and (
            (not placeholder_only) ^ is_only_operator_on_placeholder(to_copy_node)
        )

    return validator


@dynamo_tensorrt_converter(
    torch.ops.aten.clone.default,
    capability_validator=lambda node, settings: not is_only_operator_on_placeholder(
        node, settings
    ),
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=False),
    supports_dynamic_shapes=True,
)
def aten_ops_clone_copy_dtype(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=False,  # force_layer=False results in better performance
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.clone.default,
    capability_validator=is_only_operator_on_placeholder,
)
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=True),
)
def aten_ops_clone_copy_placeholder(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # For clone or copy nodes where the input is also the output,
    # we need to force cast to ensure a layer is added to the TRT engine
    # since TRT engine inputs cannot also be TRT engine outputs
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=True,
    )


@dynamo_tensorrt_converter(torch.ops.aten.expand.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_expand(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.expand(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.amax.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_amax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.amax(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=[]),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.amin.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_amin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.amin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=[]),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.sum.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.sum.dim_IntList, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.prims.sum.default, supports_dynamic_shapes=True)
def aten_ops_sum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    sum_ = impl.reduce.sum(
        ctx,
        target,
        get_ir(target),
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )

    if kwargs.get("output_dtype", None) is not None:
        return impl.cast.to_copy(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            sum_,
            kwargs["output_dtype"],
            force_layer=False,  # force_layer=False results in better performance
        )
    else:
        return sum_


@dynamo_tensorrt_converter(torch.ops.aten.prod.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.prod.dim_int, supports_dynamic_shapes=True)
def aten_ops_prod(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.prod(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.max.default,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.max.dim,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_max(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.max(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, replacement=None),
        keepdim=args_bounds_check(args, 2, replacement=False),
        return_indices=(target == torch.ops.aten.max.dim),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.min.default,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.min.dim,
    capability_validator=one_user_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_min(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.min(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, replacement=None),
        keepdim=args_bounds_check(args, 2, replacement=False),
        return_indices=(target == torch.ops.aten.min.dim),
    )


@dynamo_tensorrt_converter(torch.ops.aten.mean.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.mean.dim, supports_dynamic_shapes=True)
def aten_ops_mean(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.mean(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.exp.default, supports_dynamic_shapes=True)
def aten_ops_exp(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.exp(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.expm1.default, supports_dynamic_shapes=True)
def aten_ops_expm1(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.expm1(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log.default, supports_dynamic_shapes=True)
def aten_ops_log(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log2.default, supports_dynamic_shapes=True)
def aten_ops_log2(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log2(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log10.default, supports_dynamic_shapes=True)
def aten_ops_log10(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log10(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log1p.default)
def aten_ops_log1p(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log1p(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sqrt.default, supports_dynamic_shapes=True)
def aten_ops_sqrt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sqrt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.reciprocal.default, supports_dynamic_shapes=True
)
def aten_ops_recip(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.recip(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.abs.default, supports_dynamic_shapes=True)
def aten_ops_abs(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.abs(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sin.default, supports_dynamic_shapes=True)
def aten_ops_sin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cos.default, supports_dynamic_shapes=True)
def aten_ops_cos(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cos(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tan.default, supports_dynamic_shapes=True)
def aten_ops_tan(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.tan(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sinh.default, supports_dynamic_shapes=True)
def aten_ops_sinh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sinh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cosh.default, supports_dynamic_shapes=True)
def aten_ops_cosh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cosh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asin.default, supports_dynamic_shapes=True)
def aten_ops_asin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acos.default, supports_dynamic_shapes=True)
def aten_ops_acos(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acos(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atan.default, supports_dynamic_shapes=True)
def aten_ops_atan(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atan(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asinh.default, supports_dynamic_shapes=True)
def aten_ops_asinh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asinh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acosh.default, supports_dynamic_shapes=True)
def aten_ops_acosh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acosh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atanh.default, supports_dynamic_shapes=True)
def aten_ops_atanh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atanh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atan2.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
    }
)
def aten_ops_atan2(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.atan2(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atan2.out, supports_dynamic_shapes=True)
def aten_ops_atan2_out(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input, other = args[0], args[1]
    # out = kwargs.get("out"),

    out_return = impl.elementwise.atan2(ctx, target, SourceIR.ATEN, name, input, other)

    return out_return


@dynamo_tensorrt_converter(torch.ops.aten.ceil.default, supports_dynamic_shapes=True)
def aten_ops_ceil(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.ceil(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.floor.default, supports_dynamic_shapes=True)
def aten_ops_floor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.floor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.logical_not.default, supports_dynamic_shapes=True
)
def aten_ops_logical_not(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.logical_not(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.sym_not, supports_dynamic_shapes=True)
def aten_ops_sym_not(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sym_not(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sign.default, supports_dynamic_shapes=True)
def aten_ops_sign(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sign(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.round.default, supports_dynamic_shapes=True)
def aten_ops_round(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.round(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.isinf.default, supports_dynamic_shapes=True)
def aten_ops_isinf(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.isinf(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.isnan.default, supports_dynamic_shapes=True)
def aten_ops_isnan(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.isnan(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(operator.add, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.add.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.add.Scalar, supports_dynamic_shapes=True)
def aten_ops_add(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            other,
            alpha,
        )

    return impl.elementwise.add(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        other,
    )


@dynamo_tensorrt_converter(operator.mul, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.mul.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.mul.Scalar, supports_dynamic_shapes=True)
def aten_ops_mul(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.mul(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.maximum.default, supports_dynamic_shapes=True)
def aten_ops_maximum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.max(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.minimum.default, supports_dynamic_shapes=True)
def aten_ops_minimum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.min(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(operator.sub, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.sub.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.sub.Scalar, supports_dynamic_shapes=True)
def aten_ops_sub(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            ctx,
            target,
            SourceIR.ATEN,
            name + "_alpha",
            other,
            alpha,
        )

    return impl.elementwise.sub(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        other,
    )


@dynamo_tensorrt_converter(operator.truediv, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor_mode, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar_mode, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.prims.div.default, supports_dynamic_shapes=True)
def aten_ops_div(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    rounding_mode = kwargs.get("rounding_mode")

    if rounding_mode is None:
        return impl.elementwise.div(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "floor":
        return impl.elementwise.floor_divide(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "trunc":
        return impl.elementwise.trunc_div(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    else:
        raise RuntimeError(
            f"Target {target} does not support rounding mode {rounding_mode}"
        )


@dynamo_tensorrt_converter(operator.pow, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(
    torch.ops.aten.pow.Tensor_Tensor, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(torch.ops.aten.pow.Scalar, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(
    torch.ops.aten.pow.Tensor_Scalar, supports_dynamic_shapes=True
)
def aten_ops_pow(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.pow(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.floor_divide.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.floor_divide.Scalar, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(operator.floordiv, supports_dynamic_shapes=True)
def aten_ops_floor_div(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.floor_divide(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.logical_and.default, supports_dynamic_shapes=True
)
def aten_ops_logical_and(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_and(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.logical_or.default, supports_dynamic_shapes=True
)
def aten_ops_logical_or(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_or(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.logical_xor.default, supports_dynamic_shapes=True
)
def aten_ops_logical_xor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_xor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def bitwise_type_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    supported_type = [torch.bool, bool]

    tensor_targets = [
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Tensor,
    ]
    scalar_targets = [
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_xor.Scalar,
    ]
    scalar_tensor_targets = [
        torch.ops.aten.bitwise_and.Scalar_Tensor,
        torch.ops.aten.bitwise_or.Scalar_Tensor,
        torch.ops.aten.bitwise_xor.Scalar_Tensor,
    ]
    if node.target in tensor_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        lhs_meta = lhs_val.meta.get("tensor_meta")
        rhs_meta = rhs_val.meta.get("tensor_meta")
        if lhs_meta is None or rhs_meta is None:
            return False
        return lhs_meta.dtype in supported_type and rhs_meta.dtype in supported_type

    elif node.target in scalar_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        lhs_meta = lhs_val.meta.get("tensor_meta")
        if lhs_meta is None:
            return False
        return lhs_meta.dtype in supported_type and isinstance(rhs_val, bool)

    elif node.target in scalar_tensor_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        rhs_meta = rhs_val.meta.get("tensor_meta")
        if rhs_meta is None:
            return False
        return isinstance(lhs_val, bool) and rhs_meta.dtype in supported_type

    else:
        return False


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Scalar,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Scalar_Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_bitwise_and(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_and(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Scalar,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Scalar_Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_bitwise_or(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_or(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Scalar,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Scalar_Tensor,
    capability_validator=bitwise_type_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_bitwise_xor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_xor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def bitwise_not_type_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    val = node.args[0]
    val_meta = val.meta.get("tensor_meta")

    if val_meta is None:
        return False

    supported_type = [torch.bool, bool]
    return val_meta.dtype in supported_type


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_not.default,
    capability_validator=bitwise_not_type_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_bitwise_not(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.bitwise_not(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.eq.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.eq.Scalar, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(operator.eq, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_eq(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.eq(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ne.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.ne.Scalar, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_ne(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.ne(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.gt.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.gt.Scalar, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_gt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.gt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ge.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.ge.Scalar, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_ge(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.ge(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.lt.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.lt.Scalar, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_lt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.lt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.le.Tensor, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.le.Scalar, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_le(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.le(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.convolution.default,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (np.ndarray, torch.Tensor, TRTTensor),
        2: (np.ndarray, torch.Tensor, TRTTensor),
    }
)
def aten_ops_convolution(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    is_transposed = args[6]
    is_conv1d = len(args[0].shape) == 3
    if not is_transposed:
        return impl.conv.convNd(
            ctx,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_conv1d=is_conv1d,
            input=args[0],
            weight=args[1],
            bias=args_bounds_check(args, 2, None),
            stride=args[3],
            padding=args[4],
            dilation=args[5],
            groups=args[8],
        )
    else:
        return impl.deconv.deconvNd(
            ctx,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_deconv1d=is_conv1d,
            input=args[0],
            weight=args[1],
            bias=args_bounds_check(args, 2, None),
            stride=args[3],
            padding=args[4],
            dilation=args[5],
            output_padding=args[7],
            groups=args[8],
        )


@dynamo_tensorrt_converter(torch.ops.aten._cdist_forward.default)
def aten_ops_cdist_forward(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.cdist_forward(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        x1=args[0],
        x2=args[1],
        p=args[2],
        compute_mode=args_bounds_check(args, 3, None),
    )


def avg_pool_param_validator(
    pool_node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    ceil_mode = args_bounds_check(pool_node.args, 4, False)
    divisor_override = args_bounds_check(pool_node.args, 6)

    if ceil_mode is not False:
        _LOGGER.debug(
            f"Currently we don't support specifying ceil_mode, got ceil_mode={ceil_mode}."
        )
        return False

    if divisor_override is not None:
        _LOGGER.debug(
            f"Currently we don't support divisor_override, got divisor_override={divisor_override}."
        )
        return False

    return True


# Note: AvgPool1d uses avg_pool2d as it converts to 2D first.
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool1d.default,
    capability_validator=avg_pool_param_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool2d.default,
    capability_validator=avg_pool_param_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool3d.default,
    capability_validator=avg_pool_param_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_avg_pool(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.avg_poolNd(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        kernel_size=args[1],
        stride=args_bounds_check(args, 2, replacement=[]),
        padding=args_bounds_check(args, 3, replacement=0),
        ceil_mode=args_bounds_check(args, 4, replacement=False),
        count_include_pad=args_bounds_check(args, 5, replacement=True),
        divisor_override=args_bounds_check(args, 6, replacement=None),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.adaptive_avg_pool1d.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_adaptive_avg_pool1d(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.adaptive_avg_pool1d(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        output_size=args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.adaptive_avg_pool2d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten._adaptive_avg_pool2d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.adaptive_avg_pool3d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten._adaptive_avg_pool3d.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_adaptive_avg_poolNd(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.adaptive_avg_poolNd(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        output_size=args[1],
    )


def topk_validator(node: Node, settings: Optional[CompilationSettings] = None) -> bool:
    k = node.args[1]
    return topk_sort_validator(k)


def sort_validator(node: Node, settings: Optional[CompilationSettings] = None) -> bool:
    meta_data = node.args[0].meta.get("tensor_meta")
    if meta_data is None:
        return False
    shape = meta_data.shape
    dim = node.args[1]
    dim = get_positive_dim(dim, len(shape))
    k = shape[dim]
    if not isinstance(k, int):
        return False
    return topk_sort_validator(k)


def topk_sort_validator(k: int) -> bool:

    # topk layer supports dynamic k value but we cannot determine supported dynamic topk value at
    # compile time.
    if k == DYNAMIC_DIM or not isinstance(k, int):
        _LOGGER.warning(
            "[top_k validator] It's not expected for k to be a dynamic or data-dependent value. aten::topk will run in PyTorch"
        )
        return False

    if k > 3840:
        _LOGGER.warning(
            f"[top_k validator] Currently only topk values up to 3840 are supported, got k={k}. Therefore, aten::topk will run in PyTorch"
        )
        return False
    return True


def max_pool_param_validator(
    pool_node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    dilation = args_bounds_check(pool_node.args, 4, 1)
    ceil_mode = args_bounds_check(pool_node.args, 5, False)

    if dilation != 1:
        _LOGGER.debug(f"Currently we don't support dilation, got dilation={dilation}.")
        return False

    if ceil_mode is not False:
        _LOGGER.debug(
            f"Currently we don't support specifying ceil_mode, got ceil_mode={ceil_mode}."
        )
        return False

    return True


# Note: MaxPool1d uses max_pool2d as it converts to 2D first.
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool1d.default,
    capability_validator=max_pool_param_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool2d.default,
    capability_validator=max_pool_param_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool3d.default,
    capability_validator=max_pool_param_validator,
    supports_dynamic_shapes=True,
)
def aten_ops_max_pool(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.max_poolNd(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        kernel_size=args[1],
        stride=args_bounds_check(args, 2, replacement=[]),
        padding=args_bounds_check(args, 3, replacement=0),
        dilation=args_bounds_check(args, 4, replacement=1),
        ceil_mode=args_bounds_check(args, 5, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.reshape.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(
    torch.ops.aten._reshape_copy.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(torch.ops.aten.view.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_reshape(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shuffle.reshape(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        shape=args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.pixel_shuffle.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_pixel_shuffle(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shuffle.pixel_shuffle(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.pixel_unshuffle.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_pixel_unshuffle(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shuffle.pixel_unshuffle(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.resize_.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_resize(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shuffle.resize(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        sizes=args[1],
    )


@enforce_tensor_types({0: (TRTTensor,)})
@dynamo_tensorrt_converter(torch.ops.aten.argmax.default, supports_dynamic_shapes=True)
def aten_ops_argmax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.argmax(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1),
        keep_dim=args_bounds_check(args, 2, False),
    )


@enforce_tensor_types({0: (TRTTensor,)})
@dynamo_tensorrt_converter(torch.ops.aten.argmin.default, supports_dynamic_shapes=True)
def aten_ops_argmin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.argmin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1),
        keep_dim=args_bounds_check(args, 2, False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.addmm.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (np.ndarray, torch.Tensor, TRTTensor),
        2: (np.ndarray, torch.Tensor, TRTTensor),
    }
)
def aten_ops_addmm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.addmm.addmm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        beta=kwargs.get("beta", 1),
        alpha=kwargs.get("alpha", 1),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.constant_pad_nd.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_constant_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.constant_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.reflection_pad1d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.reflection_pad2d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.reflection_pad3d.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_reflection_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.reflection_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.replication_pad1d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.replication_pad2d.default, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.replication_pad3d.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_replication_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.replication_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._pad_circular.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_circular_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.circular_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.pad.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.pad(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        pad=args[1],
        mode=args_bounds_check(args, 2, "constant"),
        value=args_bounds_check(args, 3, None),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_nearest1d.vec, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_nearest2d.vec, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_nearest3d.vec, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_upsample_nearest(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.upsample.upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 2),
        mode="nearest",
        align_corners=False,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_linear1d.vec, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_bilinear2d.vec, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_trilinear3d.vec, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_upsample_linear(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.upsample.upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 3),
        mode="linear",
        align_corners=args[2],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.upsample_bicubic2d.vec, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_upsample_bicubic2d(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.upsample.upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 3),
        mode="bicubic",
        align_corners=args[2],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.topk.default,
    capability_validator=topk_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_topk(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.topk(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        k=args[1],
        dim=args_bounds_check(args, 2, -1),
        largest=args_bounds_check(args, 3, True),
        sorted=args_bounds_check(args, 4, True),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.sort.default,
    capability_validator=sort_validator,
    supports_dynamic_shapes=True,
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_sort(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.sort(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, -1),
        descending=args_bounds_check(args, 2, False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.trunc.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_trunc(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.trunc(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.copy.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        1: (TRTTensor,),
    }
)
def aten_ops_copy(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    src = args[1]
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        src,
        src.dtype,
        force_layer=True,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.remainder.Scalar, supports_dynamic_shapes=True
)
@dynamo_tensorrt_converter(
    torch.ops.aten.remainder.Tensor, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_remainder(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.remainder(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.any.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.any.dim, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.any.dims, supports_dynamic_shapes=True)
def aten_ops_any(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.any(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._pdist_forward.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_pdist(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.pdist(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.flip.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_flip(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.flip(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def zero_diag_size_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    meta = node.args[0].meta.get("tensor_meta")
    if meta:
        input_shape = meta.shape
    else:
        _LOGGER.warning(
            "Meta information of input is missing. Unable to validate diagonal size, falling back to PyTorch operation."
        )
        return False

    if len(node.args) == 1:
        offset, dim1, dim2 = 0, 0, 1
    elif len(node.args) == 2:
        offset, dim1, dim2 = node.args[1], 0, 1
    else:
        offset, dim1, dim2 = (
            node.args[1],
            node.args[2],
            node.args[3],
        )
    num_dims = len(input_shape)

    # Adjust dimensions to be positive and canonicalize
    dim1 = get_positive_dim(dim1, num_dims)
    dim2 = get_positive_dim(dim2, num_dims)

    if offset >= 0:
        diag_size = max(min(input_shape[dim1], input_shape[dim2] - offset), 0)
    else:
        diag_size = max(min(input_shape[dim1] + offset, input_shape[dim2]), 0)

    if diag_size == 0:
        _LOGGER.debug(
            "Diagonal size is zero, resulting in an empty tensor which is not supported for this operation."
        )
        return False
    else:
        return True


@dynamo_tensorrt_converter(
    torch.ops.aten.diagonal.default, capability_validator=zero_diag_size_validator
)
def aten_ops_diagonal(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.diagonal(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=0),
        args_bounds_check(args, 2, replacement=0),
        args_bounds_check(args, 3, replacement=1),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.scalar_tensor.default, supports_dynamic_shapes=True
)
def aten_ops_scalar_tensor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.scalar_tensor(
        ctx, target, SourceIR.ATEN, name, args[0], dtype=kwargs.get("dtype")
    )


@dynamo_tensorrt_converter(torch.ops.aten.roll.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_roll(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.permutation.roll(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, []),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.index_select.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        2: (TRTTensor,),
    }
)
def aten_ops_index_select(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.index_select(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
    )


def dropout_inference_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    train_mode = args_bounds_check(node.args, 2, None)
    if train_mode is False:
        return True
    else:  # train_mode is True or None
        _LOGGER.debug(
            "Currently only inference mode is supported for dropout operation."
        )
        return False


@dynamo_tensorrt_converter(
    torch.ops.aten.native_dropout.default,
    capability_validator=dropout_inference_validator,
)
def aten_ops_native_dropout(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.native_dropout(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, None),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._prelu_kernel.default, supports_dynamic_shapes=True
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
    }
)
def aten_ops_prelu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.prelu.prelu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.arange.start_step, supports_dynamic_shapes=True
)
def aten_ops_arange_start_step(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.arange.arange(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        start=args[0],
        end=args[1],
        step=args_bounds_check(args, 2, 1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.full.default, supports_dynamic_shapes=True)
def aten_ops_full(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.full.full(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        shape=args[0],
        fill_value=args[1],
        dtype=kwargs.get("dtype", None),
    )


# currently nonzero is not supported for tensorrt_rtx
# TODO: lan to add the nonzero support once tensorrt_rtx team has added the support
@dynamo_tensorrt_converter(
    torch.ops.aten.nonzero.default,
    supports_dynamic_shapes=True,
    requires_output_allocator=True,
)
@needs_not_tensorrt_rtx
def aten_ops_nonzero(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.nonzero(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.linear.default, supports_dynamic_shapes=True)
def aten_ops_linear(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.linear.linear(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args_bounds_check(args, 2, None),
    )
