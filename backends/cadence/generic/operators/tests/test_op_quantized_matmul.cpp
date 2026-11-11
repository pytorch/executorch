/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <optional>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/generic/operators/op_quantized_matmul.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// The kernel computes -out_multiplier / (1 << 31) * 2^out_shift, and 1 << 31 is
// INT_MIN, so the scale is +out_multiplier / 2^31 * 2^out_shift, as
// quantize_tensor_multiplier produces: 1 << 29 is 1/4.
// aot/ref_implementations.py computes -out_multiplier / 2^31, so the expected
// values come from it with -(1 << 29).
constexpr int64_t kQuarter = 1 << 29;

class GenericQuantizedMatmulAsym8uTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Byte> tf_;
};

// (X - 128) @ (Y - 128) = [[-7, 88], [-8, 8]], times 1/4, plus 100.
TEST_F(GenericQuantizedMatmulAsym8uTest, Asym8u) {
  const Tensor x = tf_.make({2, 3}, {130, 125, 140, 120, 128, 128});
  const Tensor y = tf_.make({3, 2}, {129, 127, 131, 126, 128, 135});
  Tensor out = tf_.zeros({2, 2});

  quantized_matmul_asym8uxasym8u_asym8u_out(
      context_, x, 128, y, 128, std::nullopt, kQuarter, 0, 100, false, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 2}, {98, 122, 98, 102}));
}

TEST_F(GenericQuantizedMatmulAsym8uTest, Asym8uTransposed) {
  const Tensor x = tf_.make({2, 3}, {130, 125, 140, 120, 128, 128});
  const Tensor y_t = tf_.make({2, 3}, {129, 131, 128, 127, 126, 135});
  Tensor out = tf_.zeros({2, 2});

  quantized_matmul_asym8uxasym8u_asym8u_out(
      context_, x, 128, y_t, 128, std::nullopt, kQuarter, 0, 100, true, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 2}, {98, 122, 98, 102}));
}

TEST_F(GenericQuantizedMatmulAsym8uTest, Asym8uSaturatesAtBothEnds) {
  const Tensor x = tf_.make({2, 3}, {255, 255, 255, 0, 0, 0});
  const Tensor y = tf_.make({3, 2}, {255, 0, 255, 0, 255, 0});
  Tensor out = tf_.zeros({2, 2});

  quantized_matmul_asym8uxasym8u_asym8u_out(
      context_, x, 128, y, 128, std::nullopt, kQuarter, 0, 128, false, out);

  EXPECT_TENSOR_EQ(out, tf_.make({2, 2}, {255, 0, 0, 255}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
