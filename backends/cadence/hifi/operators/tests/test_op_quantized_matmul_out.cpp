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

class HiFiQuantizedMatmulTest : public OperatorTest {
 public:
 protected:
  void quantized_matmul_out(
      const Tensor& X,
      int64_t X_zero_point,
      const Tensor& Y,
      int64_t Y_zero_point,
      const std::optional<Tensor>& bias,
      int64_t out_multiplier,
      int64_t out_shift,
      int64_t out_zero_point,
      bool transposed,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_matmul_out(
        context_,
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        output);
  }
};

// Test quantized_matmul_out with int16 activations and int8 weights
TEST_F(HiFiQuantizedMatmulTest, QuantizedMatmulInt16Test) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;

  // Simple 2D case: X [64, 33] x Y [33, 128] = output [64, 128]
  // Using simple values for testing
  Tensor X = tf_int16.ones({64, 33});
  Tensor Y = tf_int8.ones({33, 128});
  // Bias not used
  Tensor bias = tf_int32.full({128}, -30);
  Tensor output = tf_int16.zeros({64, 128});

  int64_t X_zero_point = 0;
  int64_t Y_zero_point = 0;
  int64_t out_multiplier = 1073741824; // 0.5 * 2^31
  int64_t out_shift = 0;
  int64_t out_zero_point = 0;

  quantized_matmul_out(
      X,
      X_zero_point,
      Y,
      Y_zero_point,
      bias, // pass bias tensor
      out_multiplier,
      out_shift,
      out_zero_point,
      false, // transposed
      output);

  // Verify the output is correct
  // With all ones input and weights, inner dimension is 33
  // Matmul result: 33, with out_multiplier = 0.5 * 2^31 (scales by 0.5)
  // Expected value: 33 * 0.5 = 16.5 ≈ 16
  EXPECT_EQ(output.const_data_ptr<int16_t>()[0], 16);
}

// Test quantized_matmul_out with transposed Y (int16 activations and int8
// weights)
TEST_F(HiFiQuantizedMatmulTest, QuantizedMatmulInt16TransposedTest) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;

  // Transposed case: X [64, 33] x Y^T [128, 33] = output [64, 128]
  Tensor X = tf_int16.ones({64, 33});
  Tensor Y = tf_int8.ones({128, 33}); // Transposed
  // Bias not used
  Tensor bias = tf_int32.full({128}, -30);
  Tensor output = tf_int16.zeros({64, 128});

  int64_t X_zero_point = 0;
  int64_t Y_zero_point = 0;
  int64_t out_multiplier = 1073741824; // 0.5 * 2^31
  int64_t out_shift = 0;
  int64_t out_zero_point = 0;

  quantized_matmul_out(
      X,
      X_zero_point,
      Y,
      Y_zero_point,
      bias, // pass bias tensor
      out_multiplier,
      out_shift,
      out_zero_point,
      true, // transposed
      output);

  // Verify the output is correct
  // With all ones input and weights, inner dimension is 33
  // Matmul result: 33, with out_multiplier = 0.5 * 2^31 (scales by 0.5)
  // Expected value: 33 * 0.5 = 16.5 ≈ 16
  EXPECT_EQ(output.const_data_ptr<int16_t>()[0], 16);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
