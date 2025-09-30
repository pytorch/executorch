/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/times.h>
#include <xtensa/sim.h>

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

class HiFiQuantizePerTensorTest : public OperatorTest {
 public:
 protected:
  void quantize_per_tensor_out(
      const Tensor& input,
      double scale,
      int64_t zero_point,
      __ET_UNUSED int64_t quant_min,
      __ET_UNUSED int64_t quant_max,
      ScalarType dtype,
      Tensor& out) {
    ::impl::HiFi::native::quantize_per_tensor_out(
        context_, input, scale, zero_point, quant_min, quant_max, dtype, out);
  }
};

TEST_F(HiFiQuantizePerTensorTest, ThrowKernelFailureForQuantMinMoreThanLimit) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{4};
  constexpr ScalarType kOutDtype = ScalarType::Int;
  TensorFactory<kOutDtype> tf_out;
  Tensor out = tf_out.zeros(sizes);
  // Some arbitrary values for scalar args.
  constexpr double kScale = 0.01;
  constexpr int64_t kZeroPoint = 32768;
  // quant_min and quant_max are not used in the computation.
  // However, the kernel should still throw a kernel failure error when
  // quant_min > std::numeric_limits<kOutDtype>::min() or quant_max <
  // std::numeric_limits<kOutDtype>::max().
  constexpr int64_t kQuantMin = 10;
  constexpr int64_t kQuantMax = std::numeric_limits<int32_t>::max();

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      quantize_per_tensor_out(
          tf.make(sizes, {1, 2, 3, 4}),
          kScale,
          kZeroPoint,
          kQuantMin,
          kQuantMax,
          kOutDtype,
          out));
}

TEST_F(HiFiQuantizePerTensorTest, ThrowKernelFailureForQuantMaxLessThanLimit) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{4};
  constexpr ScalarType kOutDtype = ScalarType::Int;
  TensorFactory<kOutDtype> tf_out;
  Tensor out = tf_out.zeros(sizes);
  // Some arbitrary values for scalar args.
  constexpr double kScale = 0.01;
  constexpr int64_t kZeroPoint = 32768;
  // quant_min and quant_max are not used in the computation.
  // However, the kernel should still throw a kernel failure error when
  // quant_min > std::numeric_limits<kOutDtype>::min() or quant_max <
  // std::numeric_limits<kOutDtype>::max().
  constexpr int64_t kQuantMin = std::numeric_limits<int32_t>::min();
  constexpr int64_t kQuantMax = 20;

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      quantize_per_tensor_out(
          tf.make(sizes, {1, 2, 3, 4}),
          kScale,
          kZeroPoint,
          kQuantMin,
          kQuantMax,
          kOutDtype,
          out));
}

TEST_F(HiFiQuantizePerTensorTest, CheckSingleElementIntQuantize) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{1};
  constexpr ScalarType kOutDtype = ScalarType::Int;
  TensorFactory<kOutDtype> tf_out;
  Tensor out = tf_out.zeros(sizes);
  // Some arbitrary values for scalar args.
  constexpr double kScale = 0.01;
  constexpr int64_t kZeroPoint = 32768;
  constexpr int64_t kQuantMin = std::numeric_limits<int32_t>::min();
  constexpr int64_t kQuantMax = std::numeric_limits<int32_t>::max();
  constexpr float kInputValue = 100.0f;
  constexpr int32_t kExpectedOutputValue = static_cast<int32_t>(
      static_cast<double>(kInputValue) / kScale + kZeroPoint);

  quantize_per_tensor_out(
      tf.make(sizes, {kInputValue}),
      kScale,
      kZeroPoint,
      kQuantMin,
      kQuantMax,
      kOutDtype,
      out);
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {kExpectedOutputValue}));
}

TEST_F(HiFiQuantizePerTensorTest, CheckSingleElementUInt16Quantize) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int> sizes{1};
  constexpr ScalarType kOutDtype = ScalarType::UInt16;
  TensorFactory<kOutDtype> tf_out;
  Tensor out = tf_out.zeros(sizes);
  // Some arbitrary values for scalar args.
  constexpr double kScale = 0.01;
  constexpr int64_t kZeroPoint = 32768;
  constexpr int64_t kQuantMin = std::numeric_limits<uint16_t>::min();
  constexpr int64_t kQuantMax = std::numeric_limits<uint16_t>::max();
  constexpr float kInputValue = 100.0f;
  constexpr uint16_t kExpectedOutputValue = static_cast<uint16_t>(
      static_cast<double>(kInputValue) / kScale + kZeroPoint);

  quantize_per_tensor_out(
      tf.make(sizes, {kInputValue}),
      kScale,
      kZeroPoint,
      kQuantMin,
      kQuantMax,
      kOutDtype,
      out);
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {kExpectedOutputValue}));
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
