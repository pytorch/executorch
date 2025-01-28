/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/NativeFunctions.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::string_view;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

// Note: This file is used for testing op_gelu for *portable kernel specific*.
// If your test case is generic and should be tested on all kernels, add it to
// executorch/kernels/test/op_gelu_test.cpp instead.

Tensor& op_gelu_out(const Tensor& self, string_view approximate, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::native::gelu_out(context, self, approximate, out);
}

TEST(OpGeluKernelTest, HandleInfAndNanInput) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {3, 2};

  Tensor in = tf.make(
      sizes,
      /*data=*/
      {-0.4775,
       -std::numeric_limits<float>::infinity(),
       -0.3984,
       NAN,
       std::numeric_limits<float>::infinity(),
       -0.4848});

  // Destination for the gelu.
  Tensor out = tf.zeros(sizes);

  // Run full gelu.
  op_gelu_out(in, "none", out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          sizes,
          /*data=*/
          {-0.15113, 0.0, -0.137515, NAN, INFINITY, -0.152183}));

  // Run tanh gelu appx.
  op_gelu_out(in, "tanh", out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          sizes,
          /*data=*/
          {-0.151145, 0.0, -0.137522, NAN, INFINITY, -0.152199}));
}
