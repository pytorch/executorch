# mypy: disallow-untyped-decorators=False

import logging
from typing import Dict, Sequence, Tuple, Union

import tensorrt as trt
from torch.fx.node import Argument, Target
from torch_tensorrt._features import needs_trtllm_for_nccl
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    tensorrt_fused_nccl_all_gather_op,
    tensorrt_fused_nccl_reduce_scatter_op,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


@needs_trtllm_for_nccl
@dynamo_tensorrt_converter(tensorrt_fused_nccl_all_gather_op)
def fused_nccl_gather(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    return impl.nccl_ops.nccl_gather(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        [args[0]],
    )


@needs_trtllm_for_nccl
@dynamo_tensorrt_converter(tensorrt_fused_nccl_reduce_scatter_op)
def fused_nccl_reduce_scatter(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    return impl.nccl_ops.nccl_reduce_scatter(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        [args[0]],
    )
