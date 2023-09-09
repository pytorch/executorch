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

    Dim = "Dim"
    DimOpt = "Dim?"
    DimList = "Dim[]"
    DimListOpt = "Dim[]?"

    Shape = "Shape"
    Stride = "Stride"
    Index = "Index"
    IndexOpt = "Index?"
    Length = "Length"
    LengthList = "Length[]"

    Param = "Param"
    Float = "Float"
    FloatOpt = "Float?"
    MemoryFormat = "MemoryFormat"

    Bool = "Bool"
    Keepdim = "Keepdim"

    def is_tensor(self):
        return self in [ArgType.Tensor, ArgType.TensorOpt]

    def is_tensor_list(self):
        return self in [ArgType.TensorList, ArgType.TensorOptList]

    def is_scalar(self):
        return self in [ArgType.Scalar, ArgType.ScalarOpt]

    def is_scalar_type(self):
        return self in [ArgType.ScalarType, ArgType.ScalarTypeOpt]

    def is_dim(self):
        return self in [ArgType.Dim, ArgType.DimOpt]

    def is_dim_list(self):
        return self in [ArgType.DimList, ArgType.DimListOpt]

    def is_shape(self):
        return self in [ArgType.Shape]

    def is_stride(self):
        return self in [ArgType.Stride]

    def is_index(self):
        return self in [ArgType.Index, ArgType.IndexOpt]

    def is_length(self):
        return self in [ArgType.Length]

    def is_length_list(self):
        return self in [ArgType.LengthList]

    def is_keepdim(self):
        return self in [ArgType.Keepdim]

    def is_param(self):
        return self in [
            ArgType.Param,
            ArgType.Float,
            ArgType.FloatOpt,
            ArgType.MemoryFormat,
        ]

    def is_bool(self):
        return self in [ArgType.Bool, ArgType.Keepdim]

    def is_float(self):
        return self in [ArgType.Float, ArgType.FloatOpt]

    def is_optional(self):
        return self in [
            ArgType.TensorOpt,
            ArgType.ScalarOpt,
            ArgType.ScalarTypeOpt,
            ArgType.DimOpt,
            ArgType.DimListOpt,
            ArgType.FloatOpt,
            ArgType.IndexOpt,
        ]

    def is_list(self):
        return self in [
            ArgType.TensorList,
            ArgType.TensorOptList,
            ArgType.DimList,
            ArgType.DimListOpt,
            ArgType.LengthList,
        ]

    def has_dtype(self):
        return (
            self.is_tensor()
            or self.is_tensor_list()
            or self.is_scalar()
            or self.is_scalar_type()
        )

    def has_shape(self):
        return self.is_tensor() or self.is_tensor_list()

    def has_length(self):
        return self.is_list()

    def is_shape_relevant(self):
        return (
            self.is_dim()
            or self.is_dim_list()
            or self.is_shape()
            or self.is_stride()
            or self.is_index()
            or self.is_length()
            or self.is_length_list()
            or self.is_keepdim()
        )
