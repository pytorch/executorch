#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.backends.apple.mps.utils.graph_bindings import graph_bindings


def get_mps_data_type(dtype):
    scalar_type_to_mps_dtype = {
        "torch.float32": graph_bindings.MPSDataTypeFloat32,
        "torch.float16": graph_bindings.MPSDataTypeFloat16,
        "torch.int32": graph_bindings.MPSDataTypeInt32,
        "torch.int64": graph_bindings.MPSDataTypeInt64,
        "torch.int16": graph_bindings.MPSDataTypeInt16,
        "torch.int8": graph_bindings.MPSDataTypeInt8,
        "torch.qint8": graph_bindings.MPSDataTypeInt8,
        "torch.uint8": graph_bindings.MPSDataTypeUInt8,
        "torch.quint8": graph_bindings.MPSDataTypeUInt8,
        "torch.bool": graph_bindings.MPSDataTypeBool,
        torch.float32: graph_bindings.MPSDataTypeFloat32,
        torch.float16: graph_bindings.MPSDataTypeFloat16,
        torch.int32: graph_bindings.MPSDataTypeInt32,
        torch.int64: graph_bindings.MPSDataTypeInt64,
        torch.int16: graph_bindings.MPSDataTypeInt16,
        torch.int8: graph_bindings.MPSDataTypeInt8,
        torch.qint8: graph_bindings.MPSDataTypeInt8,
        torch.uint8: graph_bindings.MPSDataTypeUInt8,
        torch.quint8: graph_bindings.MPSDataTypeUInt8,
        torch.bool: graph_bindings.MPSDataTypeBool,
    }

    try:
        return scalar_type_to_mps_dtype[dtype]
    except KeyError:
        raise AssertionError(f"Invalid data type: {dtype}")
