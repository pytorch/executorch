# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

####################
## ATen C++ Types ##
####################

AT_INT_ARRAY_REF = "at::IntArrayRef"
AT_SCALAR = "at::Scalar"
AT_TENSOR = "at::Tensor"
AT_TENSOR_LIST = "at::TensorList"
BOOL = "bool"
DOUBLE = "double"
INT = "int64_t"
OPT_AT_DOUBLE_ARRAY_REF = "::std::optional<at::ArrayRef<double>>"
OPT_AT_INT_ARRAY_REF = "at::OptionalIntArrayRef"
OPT_AT_TENSOR = "::std::optional<at::Tensor>"
OPT_BOOL = "::std::optional<bool>"
OPT_INT64 = "::std::optional<int64_t>"
OPT_DEVICE = "::std::optional<at::Device>"
OPT_LAYOUT = "::std::optional<at::Layout>"
OPT_MEMORY_FORMAT = "::std::optional<at::MemoryFormat>"
OPT_SCALAR_TYPE = "::std::optional<at::ScalarType>"
STRING = "std::string_view"
OLD_STRING = "c10::string_view"
TWO_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor>"
THREE_TENSOR_TUPLE = "::std::tuple<at::Tensor,at::Tensor,at::Tensor>"
TENSOR_VECTOR = "::std::vector<at::Tensor>"
