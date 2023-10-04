# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ArgType(str, Enum):
    Tensor = "Tensor"
    TensorOpt = "Tensor?"
    TensorList = "Tensor[]"
    TensorOptList = "Tensor?[]"
    Scalar = "Scalar"
    ScalarOpt = "Scalar?"
    ScalarType = "ScalarType"
    ScalarTypeOpt = "ScalarType?"
    Param = "Param"

    def is_tensor(self):
        return self in [ArgType.Tensor, ArgType.TensorOpt]

    def is_tensor_list(self):
        return self in [ArgType.TensorList, ArgType.TensorOptList]

    def is_scalar(self):
        return self in [ArgType.Scalar, ArgType.ScalarOpt]

    def is_scalar_type(self):
        return self in [ArgType.ScalarType, ArgType.ScalarTypeOpt]

    def is_optional(self):
        return self in [
            ArgType.TensorOpt,
            ArgType.ScalarOpt,
            ArgType.ScalarTypeOpt,
        ]

    def has_dtype(self):
        return (
            self.is_tensor()
            or self.is_tensor_list()
            or self.is_scalar()
            or self.is_scalar_type()
        )
