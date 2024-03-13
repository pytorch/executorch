# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

# Following dict is a mapping between the supported non-quantized tensor dtype to its ScalarType name.
#
# The ScalarType here are from the union between supported scalartypes for whole executorch
# runtime system (https://fburl.com/code/si7fnrxr) and executorch tensor_impl aims to
# support (https://fburl.com/code/7119zvu0).
#
# Regular tensor here means non quantized tensor.
#
# The keys are corresponding torch dtypes.
regular_tensor_dtypes_to_str = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.float: "Float",
    torch.double: "Double",
}

regular_tensor_str_to_dtypes = {
    value: key for key, value in regular_tensor_dtypes_to_str.items()
}


# The following dict is a mapping between the supported quantized tensor dtype to its ScalarType name.
# These are two quantized dtypes currently used, but support for them is deprecated and will be removed
# when early internal users are migrated off.
quantized_tensor_dtypes_to_str = {
    torch.qint8: "QINT8",
    torch.quint8: "QUINT8",
}

quantized_tensor_str_to_dtypes = {
    value: key for key, value in quantized_tensor_dtypes_to_str.items()
}
