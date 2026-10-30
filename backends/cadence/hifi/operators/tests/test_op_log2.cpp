/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/hifi/operators/operators.h>

namespace impl {
namespace HiFi {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class HiFiLog2Test : public OperatorTest {
 protected:
  Tensor& log2_out(const Tensor& in, Tensor& out) {
    return ::impl::HiFi::native::log2_out(context_, in, out);
  }
};

// Log-power style values across 120 octaves at odd lengths, within 2 ULP of
// a double-precision log2.
TEST_F(HiFiLog2Test, WideRangeWithinTwoUlp) {
  TensorFactory<ScalarType::Float> tf;
  for (const int length : {1, 2, 3, 128, 641}) {
    std::vector<float> x(length);
    for (int i = 0; i < length; ++i) {
      const double e = length == 1 ? 3.0 : -60.0 + 120.0 * i / (length - 1);
      x[i] = static_cast<float>(std::exp2(e) * (1.0 + 0.37 * (i % 11) / 11.0));
    }
    Tensor out = tf.zeros({length});
    log2_out(tf.make({length}, x), out);
    const float* y = out.const_data_ptr<float>();
    for (int i = 0; i < length; ++i) {
      const double want = std::log2(static_cast<double>(x[i]));
      const float w = static_cast<float>(want);
      const double ulp =
          std::nextafter(std::fabs(w), std::numeric_limits<float>::infinity()) -
          std::fabs(w);
      EXPECT_LE(std::fabs(y[i] - want), 2.0 * ulp)
          << "length " << length << " x " << x[i];
    }
  }
}

// Zero, negative, infinite and NaN inputs follow std::log2.
TEST_F(HiFiLog2Test, SpecialValues) {
  TensorFactory<ScalarType::Float> tf;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> x = {0.0f, -0.0f, -1.0f, inf, nan, 1.0f};
  Tensor out = tf.zeros({6});
  log2_out(tf.make({6}, x), out);
  const float* y = out.const_data_ptr<float>();
  // Compare bit patterns for NaN, since fast-math builds fold std::isnan.
  const auto is_nan = [](float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
  };
  for (size_t i = 0; i < x.size(); ++i) {
    const float want = std::log2(x[i]);
    if (is_nan(want)) {
      EXPECT_TRUE(is_nan(y[i])) << "x " << x[i];
    } else {
      EXPECT_EQ(y[i], want) << "x " << x[i];
    }
  }
}

// In-place calls use the portable kernel, since nnlib forbids overlap.
TEST_F(HiFiLog2Test, InPlace) {
  TensorFactory<ScalarType::Float> tf;
  Tensor t = tf.make({4}, {1.0f, 2.0f, 8.0f, 0.5f});
  log2_out(t, t);
  EXPECT_TENSOR_CLOSE(t, tf.make({4}, {0.0f, 1.0f, 3.0f, -1.0f}));
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
