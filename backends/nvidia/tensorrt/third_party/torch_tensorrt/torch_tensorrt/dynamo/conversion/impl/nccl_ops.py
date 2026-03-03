import os
from enum import IntEnum, IntFlag, auto
from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import set_layer_name


# class for AllReduce
class AllReduceStrategy(IntEnum):
    """Warning: actual definition is in kernels/customAllReduceKernels.h.

    They must be kept in sync.
    """

    NCCL = 0
    ONESHOT = 1
    TWOSHOT = 2
    AUTO = 3


class AllReduceConfig(IntFlag):
    """Warning: actual definition is in kernels/customAllReduceKernels.h.

    They must be kept in sync
    """

    USE_MEMCPY = auto()
    PUSH_MODE = auto()


def nccl_gather(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
) -> trt.ITensor:
    allgather_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "AllGather", "1", "tensorrt_llm"
    )
    assert allgather_plg_creator is not None
    _world_size = os.environ.get("WORLD_SIZE")
    if _world_size is not None:
        world_size = int(_world_size)
    else:
        raise RuntimeError(
            "The WORLD_SIZE env variable is not set in distributed environment"
        )
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )
    p_dtype = trt.float32
    pf_type = trt.PluginField(
        "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
    )
    pfc = trt.PluginFieldCollection([group, pf_type])
    allgather = allgather_plg_creator.create_plugin("allgather", pfc)
    layer = ctx.net.add_plugin_v2(plug_inputs, allgather)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def nccl_reduce_scatter(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
) -> trt.ITensor:
    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "ReduceScatter", "1", "tensorrt_llm"
    )

    assert allreduce_plg_creator is not None

    counter = 0
    strategy = AllReduceStrategy.NCCL
    config = AllReduceConfig(0)
    _world_size = os.environ.get("WORLD_SIZE")
    if _world_size is not None:
        world_size = int(_world_size)
    else:
        raise RuntimeError(
            "The WORLD_SIZE env variable is not set in distributed environment"
        )
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )

    p_dtype = trt.float32
    pf_dtype = trt.PluginField(
        "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
    )
    pfc = [group, pf_dtype]
    p_strategy = trt.PluginField(
        "strategy", np.array([int(strategy)], np.int8), trt.PluginFieldType.INT8
    )
    pfc.append(p_strategy)
    p_config = trt.PluginField(
        "config", np.array([int(config)], np.int8), trt.PluginFieldType.INT8
    )
    pfc.append(p_config)
    p_counter = trt.PluginField(
        "counter", np.array([counter], np.int32), trt.PluginFieldType.INT32
    )
    pfc.append(p_counter)

    pfc = trt.PluginFieldCollection(pfc)
    ar_plug = allreduce_plg_creator.create_plugin("allreduce", pfc)

    layer = ctx.net.add_plugin_v2(plug_inputs, ar_plug)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
