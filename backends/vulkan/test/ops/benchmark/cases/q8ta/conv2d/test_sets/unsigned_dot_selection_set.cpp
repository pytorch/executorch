// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/im2col/Q8taConv2dIm2Col.h>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

void check_unsigned_dot_selection() {
  const vkapi::Adapter& adapter = *vkcompute::api::context()->adapter_ptr();
  const bool prefers_unsigned_dot =
      adapter.accelerates_unsigned_packed4x8_dot() &&
      !adapter.accelerates_signed_packed4x8_dot();
  VK_CHECK_COND(
      can_use_unsigned_pw_dot(adapter, kMaxUnsignedDotAccumulatorBytes) ==
      prefers_unsigned_dot);
  VK_CHECK_COND(
      !can_use_unsigned_pw_dot(adapter, kMaxUnsignedDotAccumulatorBytes + 1));
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "unsigned_dot_selection") {
  TestCaseSet set;
  set.custom_test = check_unsigned_dot_selection;
  return set;
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
