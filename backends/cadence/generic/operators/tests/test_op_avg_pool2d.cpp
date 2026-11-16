/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_avg_pool2d.h>

#include <array>
#include <optional>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Expected values are torch.nn.functional.avg_pool2d's for the same arguments.
class GenericAvgPool2dTest : public OperatorTest {
 protected:
  Tensor& avg_pool2d_out(
      const Tensor& input,
      std::array<int64_t, 2> kernel_size,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> padding,
      bool count_include_pad,
      std::optional<int64_t> divisor_override,
      Tensor& out) {
    return impl::generic::native::avg_pool2d_out(
        context_,
        input,
        IntArrayRef(kernel_size.data(), kernel_size.size()),
        IntArrayRef(stride.data(), stride.size()),
        IntArrayRef(padding.data(), padding.size()),
        /*ceil_mode=*/true,
        count_include_pad,
        divisor_override,
        /*in_zero_point_t=*/std::nullopt,
        /*channel_last=*/false,
        out);
  }

  TensorFactory<ScalarType::Float> tf_;
};

// A 1x3 kernel with stride 2 over 6 values: ceil_mode adds a third window that
// covers only the last 2 values, so its divisor is 2, not 3.
TEST_F(GenericAvgPool2dTest, CeilModeLastWindowIncludePad) {
  Tensor input = tf_.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_.zeros({1, 1, 1, 3});

  avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 0}, true, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 1, 3}, {2.0, 4.0, 5.5}));
}

TEST_F(GenericAvgPool2dTest, CeilModeLastWindowExcludePad) {
  Tensor input = tf_.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_.zeros({1, 1, 1, 3});

  avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 0}, false, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 1, 3}, {2.0, 4.0, 5.5}));
}

// Padding of 1 on both sides: the first window covers 1 padding value and 2
// inputs, and the last covers 1 input, 1 padding value and 1 position past the
// padding, which counts toward no divisor.
TEST_F(GenericAvgPool2dTest, CeilModePaddedIncludePad) {
  Tensor input = tf_.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_.zeros({1, 1, 1, 4});

  avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 1}, true, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 1, 4}, {1.0, 3.0, 5.0, 3.0}));
}

TEST_F(GenericAvgPool2dTest, CeilModePaddedExcludePad) {
  Tensor input = tf_.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_.zeros({1, 1, 1, 4});

  avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 1}, false, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 1, 4}, {1.5, 3.0, 5.0, 6.0}));
}

TEST_F(GenericAvgPool2dTest, CeilModeDivisorOverride) {
  Tensor input = tf_.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_.zeros({1, 1, 1, 4});

  avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 1}, true, 2, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 1, 4}, {1.5, 4.5, 7.5, 3.0}));
}

// A 2x2 kernel with stride 2 over 3x3: the windows on the last row and column
// cover 2, 2 and 1 inputs.
TEST_F(GenericAvgPool2dTest, CeilModeTwoDimensions) {
  Tensor input = tf_.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor out = tf_.zeros({1, 1, 2, 2});

  avg_pool2d_out(input, {2, 2}, {2, 2}, {0, 0}, true, std::nullopt, out);

  EXPECT_TENSOR_EQ(out, tf_.make({1, 1, 2, 2}, {3.0, 4.5, 7.5, 9.0}));
}

// The uint8 path pads with the zero point, and the last window covers 140, the
// zero point and a position past the padding: (140 + 128) / 2 = 134.
TEST_F(GenericAvgPool2dTest, CeilModePaddedIncludePadUint8) {
  TensorFactory<ScalarType::Byte> tf_uint8;
  TensorFactory<ScalarType::Int> tf_int;
  Tensor input = tf_uint8.make({1, 1, 1, 6}, {130, 132, 134, 136, 138, 140});
  Tensor out = tf_uint8.zeros({1, 1, 1, 4});
  const std::array<int64_t, 2> kernel_size{1, 3}, stride{1, 2}, padding{0, 1};

  impl::generic::native::avg_pool2d_out(
      context_,
      input,
      IntArrayRef(kernel_size.data(), kernel_size.size()),
      IntArrayRef(stride.data(), stride.size()),
      IntArrayRef(padding.data(), padding.size()),
      /*ceil_mode=*/true,
      /*count_include_pad=*/true,
      /*divisor_override=*/std::nullopt,
      tf_int.make({1}, {128}),
      /*channel_last=*/false,
      out);

  EXPECT_TENSOR_EQ(out, tf_uint8.make({1, 1, 1, 4}, {130, 134, 138, 134}));
}

TEST_F(GenericAvgPool2dTest, RejectsUnsupportedDtype) {
  TensorFactory<ScalarType::Int> tf_int;
  Tensor input = tf_int.make({1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_int.zeros({1, 1, 1, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      avg_pool2d_out(input, {1, 3}, {1, 2}, {0, 0}, true, std::nullopt, out));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
