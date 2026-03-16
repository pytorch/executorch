/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/operators/op_quantized_matmul_out.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

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
  Tensor& quantized_matmul_out(
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
    return impl::HiFi::native::quantized_matmul_out(
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

  // Minimal test case: X [2, 2] x Y [2, 2] = output [2, 2]
  // Small enough to verify by hand calculation
  //
  // X (2x2):          Y (2x2):
  // 2  4              1  2
  // 6  8              1  0
  //
  // Hand calculation for matmul (before scaling):
  // (0,0): 2*1 + 4*1 = 6
  // (0,1): 2*2 + 4*0 = 4
  // (1,0): 6*1 + 8*1 = 14
  // (1,1): 6*2 + 8*0 = 12
  //
  // Raw result: [[6, 4], [14, 12]]
  // After 0.5 scaling: [[3, 2], [7, 6]]
  Tensor X = tf_int16.make({2, 2}, {2, 4, 6, 8});
  Tensor Y = tf_int8.make({2, 2}, {1, 2, 1, 0});
  Tensor bias = tf_int32.zeros({2});
  Tensor output = tf_int16.zeros({2, 2});

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
      bias,
      out_multiplier,
      out_shift,
      out_zero_point,
      false, // transposed
      output);

  Tensor expected = tf_int16.make({2, 2}, {3, 2, 7, 6});
  EXPECT_TENSOR_EQ(output, expected);
}

// Test quantized_matmul_out with transposed Y (int16 activations and int8
// weights)
TEST_F(HiFiQuantizedMatmulTest, QuantizedMatmulInt16TransposedTest) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;

  // Minimal test case with transposed Y: X [2, 2] x Y^T [2, 2] = output [2, 2]
  // Y is stored transposed, so we compute X @ Y^T
  //
  // X (2x2):          Y_stored (2x2, which is Y^T):
  // 2  4              1  1
  // 6  8              2  0
  //
  // When transposed=true, we compute X @ Y_stored^T = X @ Y
  // Y = Y_stored^T = [[1, 2], [1, 0]]
  //
  // Hand calculation for matmul (before scaling):
  // (0,0): 2*1 + 4*1 = 6
  // (0,1): 2*2 + 4*0 = 4
  // (1,0): 6*1 + 8*1 = 14
  // (1,1): 6*2 + 8*0 = 12
  //
  // Raw result: [[6, 4], [14, 12]]
  // After 0.5 scaling: [[3, 2], [7, 6]]
  Tensor X = tf_int16.make({2, 2}, {2, 4, 6, 8});
  Tensor Y = tf_int8.make({2, 2}, {1, 1, 2, 0}); // Stored as Y^T
  Tensor bias = tf_int32.zeros({2});
  Tensor output = tf_int16.zeros({2, 2});

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
      bias,
      out_multiplier,
      out_shift,
      out_zero_point,
      true, // transposed
      output);

  Tensor expected = tf_int16.make({2, 2}, {3, 2, 7, 6});
  EXPECT_TENSOR_EQ(output, expected);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
