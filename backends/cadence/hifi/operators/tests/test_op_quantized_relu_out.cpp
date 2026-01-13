/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/times.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/backends/cadence/hifi/operators/operators.h>

namespace impl {
namespace HiFi {
namespace native {
namespace {

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;
using std::optional;
using std::string_view;

class HiFiQuantizedReluTest : public OperatorTest {
 public:
 protected:
  void quantized_relu_out(
      const Tensor& input,
      const Tensor& in_zero_point,
      const int64_t out_zero_point,
      const Tensor& out_multiplier,
      const Tensor& out_shift,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_relu_out(
        context_,
        input,
        in_zero_point,
        out_zero_point,
        out_multiplier,
        out_shift,
        output);
  }
};

TEST_F(HiFiQuantizedReluTest, MultiDimensionalTest) {
  TensorFactory<ScalarType::Char> tf_chars;
  TensorFactory<ScalarType::Int> tf_ints;
  const std::vector<int32_t> sizes{2, 3, 5, 6};
  Tensor quantized_input = tf_chars.full(sizes, -128);
  Tensor quantized_output = tf_chars.full(sizes, 100);
  Tensor in_zero_point = tf_chars.full({1}, 127);
  int64_t out_zero_point = -128;
  Tensor out_multiplier = tf_ints.full({1}, 1077952640);
  Tensor out_shift = tf_ints.full({1}, 5);

  quantized_relu_out(
      quantized_input,
      in_zero_point,
      out_zero_point,
      out_multiplier,
      out_shift,
      quantized_output);

  Tensor expected_output = tf_chars.full(sizes, -128);
  EXPECT_TENSOR_EQ(quantized_output, expected_output);
}

TEST_F(HiFiQuantizedReluTest, OneDimensionalTest) {
  TensorFactory<ScalarType::Char> tf_chars;
  TensorFactory<ScalarType::Int> tf_ints;
  const std::vector<int32_t> sizes{56};
  Tensor quantized_input = tf_chars.full(sizes, -128);
  Tensor quantized_output = tf_chars.full(sizes, 100);
  Tensor in_zero_point = tf_chars.full({1}, 127);
  int64_t out_zero_point = -128;
  Tensor out_multiplier = tf_ints.full({1}, 1077952640);
  Tensor out_shift = tf_ints.full({1}, 5);

  quantized_relu_out(
      quantized_input,
      in_zero_point,
      out_zero_point,
      out_multiplier,
      out_shift,
      quantized_output);

  Tensor expected_output = tf_chars.full(sizes, -128);
  EXPECT_TENSOR_EQ(quantized_output, expected_output);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
