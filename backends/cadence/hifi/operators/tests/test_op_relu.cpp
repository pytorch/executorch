/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

using ::executorch::aten::DimOrderType;
using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::aten::StridesType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::testing::TensorFactory;

class HiFiReluTest : public OperatorTest {
 protected:
  Tensor& relu_out(const Tensor& in, Tensor& out) {
    return ::impl::HiFi::native::relu_out(context_, in, out);
  }
};

// A 1-D float view of `length` elements starting `offset` floats into `data`.
struct View {
  View(float* data, int offset, int length)
      : sizes{static_cast<SizesType>(length)},
        impl(
            ScalarType::Float,
            1,
            sizes.data(),
            data + offset,
            dim_order.data(),
            strides.data()),
        tensor(&impl) {}
  std::array<SizesType, 1> sizes;
  std::array<DimOrderType, 1> dim_order{0};
  std::array<StridesType, 1> strides{1};
  TensorImpl impl;
  Tensor tensor;
};

// Separate input and output (the nnlib path) at odd lengths, with each
// pointer misaligned by one element in turn. Sentinels around the output catch
// stores outside it.
TEST_F(HiFiReluTest, OutOfPlaceAnyLengthAndAlignment) {
  TensorFactory<ScalarType::Float> tf;
  constexpr float kSentinel = -123.0f;
  constexpr int kGuard = 2;
  for (const int in_offset : {0, 1}) {
    for (const int out_offset : {0, 1}) {
      for (const int length : {1, 2, 3, 8, 9, 33}) {
        std::vector<float> in(in_offset + length, 0.0f);
        std::vector<float> out(out_offset + length + kGuard, kSentinel);
        std::vector<float> expected = out;
        for (int i = 0; i < length; ++i) {
          in[in_offset + i] = static_cast<float>((i * 7) % 13) - 6.0f;
          expected[out_offset + i] = std::max(in[in_offset + i], 0.0f);
        }
        View in_view(in.data(), in_offset, length);
        View out_view(out.data(), out_offset, length);
        relu_out(in_view.tensor, out_view.tensor);
        const int total = static_cast<int>(out.size());
        EXPECT_TENSOR_EQ(tf.make({total}, out), tf.make({total}, expected))
            << "in offset " << in_offset << " out offset " << out_offset
            << " length " << length;
      }
    }
  }
}

// The output may alias the input, at any length and alignment.
TEST_F(HiFiReluTest, InPlaceAnyLengthAndAlignment) {
  TensorFactory<ScalarType::Float> tf;
  constexpr float kSentinel = -123.0f;
  constexpr int kGuard = 2;
  for (const int offset : {0, 1}) {
    for (const int length : {1, 2, 7, 8, 9, 33}) {
      std::vector<float> values(offset + length + kGuard, kSentinel);
      std::vector<float> expected = values;
      for (int i = 0; i < length; ++i) {
        values[offset + i] = static_cast<float>((i * 5) % 11) - 5.0f;
        expected[offset + i] = std::max(values[offset + i], 0.0f);
      }
      View view(values.data(), offset, length);
      relu_out(view.tensor, view.tensor);
      const int total = static_cast<int>(values.size());
      EXPECT_TENSOR_EQ(tf.make({total}, values), tf.make({total}, expected))
          << "offset " << offset << " length " << length;
    }
  }
}

// Non-float tensors use the portable kernel.
TEST_F(HiFiReluTest, IntUsesPortable) {
  TensorFactory<ScalarType::Int> tf;
  const Tensor in = tf.make({2, 3}, {-3, 0, 4, -1, 7, -9});
  Tensor out = tf.zeros({2, 3});
  relu_out(in, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 3}, {0, 0, 4, 0, 7, 0}));
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
