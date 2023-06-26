// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/portable/NativeFunctions.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
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

Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::native::mul_out(context, self, other, out);
}

TEST(OpMulOutKernelTest, UnhandledDtypeDies) {
  // mul_out() doesn't handle QInt8.
  // TensorFactory cannot be used with ScalarType::QInt8 since
  // torch::executor::qint8 does not have a default constructor. It must be
  // initialized with an explicit value. So, we need to manually create the
  // underlying data without default construction and then the tensors from that
  // data via TensorImpl.

  std::vector<SizesType> sizes = {2, 2};

  std::vector<torch::executor::qint8> a_data{};
  std::generate_n(std::back_inserter(a_data), 4, []() {
    return torch::executor::qint8{0};
  });
  std::vector<torch::executor::qint8> b_data(a_data);
  std::vector<torch::executor::qint8> out_data(a_data);

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
  ET_EXPECT_KERNEL_FAILURE(mul_out(a, b, out));
}
