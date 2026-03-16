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

class HiFiDequantizePerTensorTest : public OperatorTest {
 public:
 protected:
  void dequantize_per_tensor_out(
      const Tensor& input,
      double scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max,
      ScalarType dtype,
      Tensor& out) {
    return ::impl::HiFi::native::dequantize_per_tensor_out(
        context_, input, scale, zero_point, quant_min, quant_max, dtype, out);
  }
};

TEST_F(HiFiDequantizePerTensorTest, MultiDimensionalTest) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Char> tf_chars;
  const std::vector<int32_t> sizes{2, 3, 5, 6};
  Tensor quantized_tensor = tf_chars.full(sizes, -128);
  Tensor output_float = tf_float.zeros(sizes);
  double dequant_scale = 0.000244140625;
  int64_t dequant_zero_point = -128;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  dequantize_per_tensor_out(
      quantized_tensor,
      dequant_scale,
      dequant_zero_point,
      quant_min,
      quant_max,
      ScalarType::Float,
      output_float);

  EXPECT_TENSOR_EQ(output_float, tf_float.zeros(sizes));
}

TEST_F(HiFiDequantizePerTensorTest, OneDimensionalTest) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Char> tf_chars;
  const std::vector<int32_t> sizes{56};
  Tensor quantized_tensor = tf_chars.full(sizes, -128);
  Tensor output_float = tf_float.zeros(sizes);
  double dequant_scale = 0.000244140625;
  int64_t dequant_zero_point = -128;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  dequantize_per_tensor_out(
      quantized_tensor,
      dequant_scale,
      dequant_zero_point,
      quant_min,
      quant_max,
      ScalarType::Float,
      output_float);

  EXPECT_TENSOR_EQ(output_float, tf_float.zeros(sizes));
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
