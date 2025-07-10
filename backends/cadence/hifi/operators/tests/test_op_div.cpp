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

namespace cadence {
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

class HiFiDivTest : public OperatorTest {
 public:
 protected:
  Tensor& div_out_mode(
      const Tensor& a,
      const Tensor& b,
      optional<string_view> mode,
      Tensor& out) {
    return ::cadence::impl::HiFi::native::div_out_mode(
        context_, a, b, mode, out);
  }
};

// TODO: Enable once hifi div_out is fixed.
TEST_F(HiFiDivTest, DISABLED_Int32FloorDivideTest) {
  TensorFactory<ScalarType::Int> tf;
  const std::vector<int> sizes{4, 5, 6};
  Tensor out = tf.zeros(sizes);
  constexpr int32_t kNumerator = 73;
  constexpr int32_t kDenominator = 55;
  // Floor division (73 / 55) = 1.
  constexpr int32_t kExpectedResult = kNumerator / kDenominator;
  const Tensor numerator = tf.full(sizes, kNumerator);
  const Tensor denominator = tf.full(sizes, kDenominator);

  div_out_mode(numerator, denominator, "floor", out);

  EXPECT_TENSOR_EQ(out, tf.full(sizes, kExpectedResult));
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
