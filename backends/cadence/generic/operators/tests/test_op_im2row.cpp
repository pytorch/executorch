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

#include <executorch/backends/cadence/generic/operators/op_im2row.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

IntArrayRef ref(const std::array<int64_t, 2>& values) {
  return IntArrayRef(values.data(), values.size());
}

// Expected values are aot/ref_implementations.py's. The output is
// [N, out_h * out_w, C * kernel_h * kernel_w], one row per output position.
class GenericIm2rowTest : public OperatorTest {};

// A 2x2 kernel with stride 2 and padding 1 over 3x3: padded positions hold the
// zero point.
TEST_F(GenericIm2rowTest, PerTensorPadsWithTheZeroPoint) {
  TensorFactory<ScalarType::Byte> tf;
  const Tensor input = tf.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor out = tf.zeros({1, 4, 4});
  const std::array<int64_t, 2> kernel{2, 2}, dilation{1, 1}, padding{1, 1},
      stride{2, 2};

  im2row_per_tensor_out(
      context_,
      input,
      ref(kernel),
      ref(dilation),
      ref(padding),
      ref(stride),
      7,
      /*channel_last=*/false,
      out);

  EXPECT_TENSOR_EQ(
      out,
      tf.make({1, 4, 4}, {7, 7, 7, 1, 7, 7, 2, 3, 7, 4, 7, 7, 5, 6, 8, 9}));
}

// A zero-point tensor with one entry per batch pads each batch with its own.
TEST_F(GenericIm2rowTest, ZeroPointPerBatch) {
  TensorFactory<ScalarType::Char> tf;
  TensorFactory<ScalarType::Int> tf_int;
  const Tensor input = tf.make({2, 1, 1, 1}, {5, -6});
  Tensor out = tf.zeros({2, 4, 4});
  const std::array<int64_t, 2> kernel{2, 2}, dilation{1, 1}, padding{1, 1},
      stride{1, 1};

  im2row_out(
      context_,
      input,
      ref(kernel),
      ref(dilation),
      ref(padding),
      ref(stride),
      tf_int.make({2}, {3, -4}),
      /*channel_last=*/false,
      out);

  EXPECT_TENSOR_EQ(
      out, tf.make({2, 4, 4}, {3,  3,  3,  5,  3,  3,  5,  3,  3,  5,  3,
                               3,  5,  3,  3,  3,  -4, -4, -4, -6, -4, -4,
                               -6, -4, -4, -6, -4, -4, -6, -4, -4, -4}));
}

// NHWC input; the output row is laid out [C][kernel position], as for NCHW,
// and padded positions hold the zero point.
TEST_F(GenericIm2rowTest, ChannelLast) {
  TensorFactory<ScalarType::Char> tf;
  const Tensor input = tf.make({1, 2, 2, 2}, {1, 10, 2, 20, 3, 30, 4, 40});
  Tensor out = tf.zeros({1, 4, 8});
  const std::array<int64_t, 2> kernel{2, 2}, dilation{1, 1}, padding{1, 1},
      stride{2, 2};

  im2row_per_tensor_out(
      context_,
      input,
      ref(kernel),
      ref(dilation),
      ref(padding),
      ref(stride),
      7,
      /*channel_last=*/true,
      out);

  EXPECT_TENSOR_EQ(out, tf.make({1, 4, 8}, {7, 7, 7, 1,  7, 7, 7,  10, 7, 7, 2,
                                            7, 7, 7, 20, 7, 7, 3,  7,  7, 7, 30,
                                            7, 7, 4, 7,  7, 7, 40, 7,  7, 7}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
