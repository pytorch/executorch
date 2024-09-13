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
#include <algorithm>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::SizesType;
using exec_aten::StridesType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

// Note: This file is used for testing op_mul for *portable kernel specific*.
// If your test case is generic and should be tested on all kernels, add it to
// executorch/kernels/test/op_mul_test.cpp instead.

class OpMulOutKernelTest : public OperatorTest {
 protected:
  Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::native::mul_out(context_, self, other, out);
  }
};

TEST_F(OpMulOutKernelTest, UnhandledDtypeDies) {
  // mul_out() doesn't handle QInt8.
  // TensorFactory cannot be used with ScalarType::QInt8 since
  // exec_aten::qint8 does not have a default constructor. It must be
  // initialized with an explicit value. So, we need to manually create the
  // underlying data without default construction and then the tensors from that
  // data via TensorImpl.

  std::vector<SizesType> sizes = {2, 2};

  std::vector<exec_aten::qint8> a_data{};
  std::generate_n(
      std::back_inserter(a_data), 4, []() { return exec_aten::qint8{0}; });
  std::vector<exec_aten::qint8> b_data(a_data);
  std::vector<exec_aten::qint8> out_data(a_data);

  auto a_impl = torch::executor::TensorImpl(
      ScalarType::QInt8, 2, sizes.data(), a_data.data());
  auto b_impl = torch::executor::TensorImpl(
      ScalarType::QInt8, 2, sizes.data(), b_data.data());
  auto out_impl = torch::executor::TensorImpl(
      ScalarType::QInt8, 2, sizes.data(), out_data.data());

  // Two input tensors.
  Tensor a(&a_impl);
  Tensor b(&b_impl);

  // Output tensor.
  Tensor out(&out_impl);

  // Multiplying the two QInt8 tensors should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(context_, mul_out(a, b, out));
}
