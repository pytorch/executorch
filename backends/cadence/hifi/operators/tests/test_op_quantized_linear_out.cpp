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

class HiFiQuantizedLinearTest : public OperatorTest {
 public:
 protected:
  void quantized_linear_out(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      int64_t in_zero_point,
      const Tensor& weight_zero_point,
      const Tensor& out_multiplier,
      const Tensor& out_shift,
      int64_t out_zero_point,
      const optional<Tensor>& offset,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_linear_out(
        context_,
        input,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        output);
  }

  void quantized_linear_per_tensor_out(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      int64_t in_zero_point,
      int64_t weight_zero_point,
      int64_t out_multiplier,
      int64_t out_shift,
      int64_t out_zero_point,
      const optional<Tensor>& offset,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_linear_per_tensor_out(
        context_,
        input,
        weight,
        bias,
        in_zero_point,
        weight_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
        offset,
        output);
  }
};

// Test quantized_linear_out with int16 activations (asym8s)
TEST_F(HiFiQuantizedLinearTest, QuantizedLinearInt16Test) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;

  // Simple 2D case: input [2, 3] x weight [4, 3] = output [2, 4]
  // Values captured from e2e test with
  // CadenceWith16BitLinearActivationsQuantizer
  Tensor input =
      tf_int16.make({2, 3}, {-28170, -26389, -32768, -31474, -32266, -29076});
  Tensor weight = tf_int8.make(
      {4, 3}, {1, 87, -128, -114, -59, 44, -1, 127, -12, 44, -46, -29});
  Tensor bias = tf_int32.zeros({4});
  Tensor output = tf_int16.zeros({2, 4});

  int64_t in_zero_point = -29822;
  Tensor weight_zero_point = tf_int32.make({1}, {2});
  Tensor out_multiplier = tf_int32.make({1}, {2011373824});
  Tensor out_shift = tf_int32.make({1}, {-8});
  int64_t out_zero_point = -30847;
  quantized_linear_out(
      input,
      weight,
      bias,
      in_zero_point,
      weight_zero_point,
      out_multiplier,
      out_shift,
      out_zero_point,
      std::nullopt,
      output);
  // Expected output from e2e test
  Tensor expected_output = tf_int16.make(
      {2, 4}, {-28384, -32767, -29144, -30862, -31956, -29486, -31985, -30756});
  EXPECT_TENSOR_CLOSE(output, expected_output);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
