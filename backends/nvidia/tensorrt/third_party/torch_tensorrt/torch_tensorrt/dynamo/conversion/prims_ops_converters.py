import logging
from typing import Dict, Sequence, Tuple, Union

import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


# TODO: expand the scope of this converter with aten.expand implementation
def broadcast_checker(
    broadcast_node: torch.fx.Node, settings: CompilationSettings = None
) -> bool:
    # The current implementation of broadcast_in_dim can only handle unsqueeze
    return all(
        broadcast_node.args[1][i] == 1
        for i in range(len(broadcast_node.args[1]))
        if i not in broadcast_node.args[2]
    )


@dynamo_tensorrt_converter(
    torch.ops.prims.broadcast_in_dim.default, capability_validator=broadcast_checker
)
def prim_ops_broadcast_in_dim(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unsqueeze.broadcast_in_dim(
        ctx,
        target,
        SourceIR.PRIM,
        name,
        args[0],
        args[1],
        args[2],
    )
