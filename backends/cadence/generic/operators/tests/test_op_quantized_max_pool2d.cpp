/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/generic/operators/op_quantized_max_pool2d.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are aot/ref_implementations.py's.
class GenericQuantizedMaxPool2dNchwTest : public OperatorTest {
 protected:
  Tensor& max_pool(
      const Tensor& input,
      std::array<int64_t, 2> padding,
      bool ceil_mode,
      Tensor& out) {
    const std::array<int64_t, 2> kernel{2, 2}, stride{2, 2}, dilation{1, 1};
    return quantized_max_pool2d_nchw_out(
        context_,
        input,
        IntArrayRef(kernel.data(), 2),
        IntArrayRef(stride.data(), 2),
        IntArrayRef(padding.data(), 2),
        IntArrayRef(dilation.data(), 2),
        ceil_mode,
        out);
  }

  TensorFactory<ScalarType::Char> tf_;
};

TEST_F(GenericQuantizedMaxPool2dNchwTest, TwoByTwoStrideTwo) {
  const Tensor input = tf_.make(
      {1, 1, 4, 4}, {1, 5, -3, 2, 4, -8, 7, 0, -1, -2, -3, -4, 9, 6, -7, 8});
  Tensor out = tf_.zeros({1, 1, 2, 2});

  max_pool(input, {0, 0}, /*ceil_mode=*/false, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 2, 2}, {5, 7, 9, 8}));
}

// Every input is negative, so a padded position that counted as 0 would win.
TEST_F(GenericQuantizedMaxPool2dNchwTest, PaddingNeverWins) {
  const Tensor input =
      tf_.make({1, 1, 3, 3}, {-5, -6, -7, -8, -9, -10, -11, -12, -13});
  Tensor out = tf_.zeros({1, 1, 2, 2});

  max_pool(input, {1, 1}, /*ceil_mode=*/false, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 2, 2}, {-5, -6, -8, -9}));
}

// ceil_mode adds the windows that cover only the last row and column.
TEST_F(GenericQuantizedMaxPool2dNchwTest, CeilMode) {
  const Tensor input =
      tf_.make({1, 1, 3, 3}, {-5, -6, -7, -8, -9, -10, -11, -12, -13});
  Tensor out = tf_.zeros({1, 1, 2, 2});

  max_pool(input, {0, 0}, /*ceil_mode=*/true, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 2, 2}, {-5, -7, -11, -13}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
