/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/vulkan/impl/Arithmetic.h>
#include <ATen/native/vulkan/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/Functions.h>

#include <executorch/backends/vulkan/runtime/graph/ops/Arithmetic.h>

namespace at {
namespace native {
namespace vulkan {

#define DEFINE_ARITHMETIC_FN(function, op_type)                               \
  ValueRef function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                               \
        graph,                                                                \
        args[0],                                                              \
        args[1],                                                              \
        args[2],                                                              \
        arithmetic::OpType::op_type,                                          \
        args[3]);                                                             \
  }

DEFINE_ARITHMETIC_FN(add, ADD);
DEFINE_ARITHMETIC_FN(sub, SUB);
DEFINE_ARITHMETIC_FN(mul, MUL);
DEFINE_ARITHMETIC_FN(div, DIV);
DEFINE_ARITHMETIC_FN(floor_div, FLOOR_DIV);
DEFINE_ARITHMETIC_FN(pow, POW);

} // namespace vulkan
} // namespace native
} // namespace at
