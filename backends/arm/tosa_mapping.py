# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# PyTorch to Tosa mapping - simple mapping functions and multi-type extraction
# of key information. These are used by the initial compile stage which captures
# the standardised TOSA representation.
#

import serializer.tosa_serializer as ts
import torch


def map_dtype(data_type):
    unsupported = (
        torch.float64,
        torch.double,
        torch.complex64,
        torch.cfloat,
        torch.complex128,
        torch.cdouble,
        torch.uint8,
        torch.int64,
        torch.long,
    )

    dmap = {
        torch.float32: ts.DType.FP32,
        torch.float: ts.DType.FP32,
        torch.float16: ts.DType.FP16,
        torch.half: ts.DType.FP16,
        torch.bfloat16: ts.DType.BF16,
        torch.int8: ts.DType.INT8,
        torch.int16: ts.DType.INT16,
        torch.short: ts.DType.INT16,
        torch.int32: ts.DType.INT32,
        torch.int: ts.DType.INT32,
        torch.bool: ts.DType.BOOL,
    }

    assert unsupported.count(data_type) == 0, "Unsupported type"
    rtype = dmap.get(data_type)
    assert rtype is not None, "Unknown type"
    return rtype


# Returns the shape and type of a node
# TODO: other types, can be
# SymInt, FakeTensor, a List[Union[FakeTensor, SymInt]], or None
def extract_tensor_meta(meta):
    assert meta.get("val") is not None
    val = meta["val"]
    if type(val) is tuple:
        # TODO: should use first concrete representation
        val = val[0]

    assert torch._subclasses.fake_tensor.FakeTensor == type(val)
    dtype = map_dtype(val.dtype)
    shape = tuple(val.size())

    if meta.get("tosa_dim_order") is not None:
        dim_order = meta["tosa_dim_order"]
    else:
        dim_order = tuple(range(len(shape)))
    return (dtype, shape, dim_order)


# Class to capture arguments and turn into tensor references for TOSA OPs
class TosaArg:
    def __process_node(self, argument):
        assert isinstance(argument, torch.fx.node.Node)
        self.name = argument.name
        self.dtype, self.shape, self.dim_order = extract_tensor_meta(argument.meta)

    def __process_list(self, argument):
        self.special = list(argument)

    def __process_number(self, argument):
        self.number = argument

    def __init__(self, argument) -> None:
        self.name = None
        self.dtype = None
        self.shape = None
        self.dim_order = None
        self.special = None

        if argument is None:
            return

        if isinstance(argument, torch.fx.node.Node):
            self.__process_node(argument)
            return
        if isinstance(argument, list):
            self.__process_list(argument)
            return
        if isinstance(argument, int):
            self.__process_number(argument)
            return
        if isinstance(argument, float):
            self.__process_number(argument)
            return

        RuntimeError(
            f"Unhandled node input argument: {argument}, of type {type(argument)}"
        )
