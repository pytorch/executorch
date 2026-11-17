/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_ring_mask_gen.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class SdpaBitwiseRingMaskGenOutTest : public OperatorTest {
 protected:
  Tensor& generate(
      const Tensor& positions,
      int64_t window,
      int64_t capacity,
      Tensor& out) {
    return sdpa_bitwise_ring_mask_gen_out(
        context_, positions, window, capacity, out);
  }
};

TEST_F(SdpaBitwiseRingMaskGenOutTest, MatchesVisibleSlotsAcrossWraps) {
  TensorFactory<ScalarType::Long> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  for (const int32_t capacity : {8, 16, 24, 2048, 4096}) {
    std::vector<int64_t> values;
    for (int64_t position = 0; position < 3 * std::min(capacity, 24);
         ++position) {
      values.push_back(position);
    }
    values.insert(
        values.end(),
        {capacity - 1,
         capacity,
         capacity + 1,
         3 * capacity + 5,
         std::numeric_limits<int64_t>::max()});
    const int32_t rows = static_cast<int32_t>(values.size());
    Tensor positions = positions_factory.make({rows}, values);
    std::vector<int64_t> windows;
    if (capacity <= 24) {
      for (int64_t window = 1; window <= capacity; ++window) {
        windows.push_back(window);
      }
    } else {
      windows = {1, 4, 8, 9, capacity - 1, capacity};
    }
    for (const int64_t window : windows) {
      SCOPED_TRACE(capacity);
      SCOPED_TRACE(window);
      Tensor out = byte_factory.full({rows, capacity / 8}, 0xA5);
      std::vector<uint8_t> expected(values.size() * capacity / 8, 0xFF);
      for (int32_t row = 0; row < rows; ++row) {
        for (int64_t age = 0; age < window && age <= values[row]; ++age) {
          const int64_t slot = (values[row] - age) % capacity;
          expected[row * capacity / 8 + slot / 8] &=
              static_cast<uint8_t>(~(1u << (slot % 8)));
        }
      }
      Tensor reference = byte_factory.make({rows, capacity / 8}, expected);
      EXPECT_EQ(&generate(positions, window, capacity, out), &out);
      EXPECT_TENSOR_EQ(out, reference);
    }
  }
}

TEST_F(SdpaBitwiseRingMaskGenOutTest, SupportsInt32Positions) {
  TensorFactory<ScalarType::Int> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  Tensor positions = positions_factory.make({3}, {0, 8, 16});
  Tensor out = byte_factory.zeros({3, 2});
  Tensor expected =
      byte_factory.make({3, 2}, {0xFE, 0xFF, 0x1F, 0xFE, 0xFE, 0x1F});
  EXPECT_EQ(&generate(positions, 4, 16, out), &out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(SdpaBitwiseRingMaskGenOutTest, SupportsEmptyPositions) {
  TensorFactory<ScalarType::Long> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  Tensor positions = positions_factory.zeros({0});
  Tensor out = byte_factory.zeros({0, 2});
  EXPECT_EQ(&generate(positions, 4, 16, out), &out);
}

#if !defined(__XTENSA__)
TEST_F(SdpaBitwiseRingMaskGenOutTest, RejectsInvalidWindowsAndCapacities) {
  TensorFactory<ScalarType::Long> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  Tensor positions = positions_factory.make({1}, {0});
  Tensor out = byte_factory.zeros({1, 2});
  for (const int64_t window : {-1, 0, 17}) {
    ET_EXPECT_KERNEL_FAILURE(context_, generate(positions, window, 16, out));
  }
  for (const int64_t capacity : {-8, 0, 15}) {
    ET_EXPECT_KERNEL_FAILURE(context_, generate(positions, 4, capacity, out));
  }
}

TEST_F(SdpaBitwiseRingMaskGenOutTest, RejectsNegativePositionsBeforeWriting) {
  TensorFactory<ScalarType::Long> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  Tensor positions = positions_factory.make({2}, {0, -1});
  Tensor out = byte_factory.full({2, 2}, 0xA5);
  Tensor expected = byte_factory.full({2, 2}, 0xA5);
  ET_EXPECT_KERNEL_FAILURE(context_, generate(positions, 4, 16, out));
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(
    SdpaBitwiseRingMaskGenOutTest,
    RejectsNegativeInt32PositionsBeforeWriting) {
  TensorFactory<ScalarType::Int> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  Tensor positions = positions_factory.make({2}, {0, -1});
  Tensor out = byte_factory.full({2, 2}, 0xA5);
  Tensor expected = byte_factory.full({2, 2}, 0xA5);
  ET_EXPECT_KERNEL_FAILURE(context_, generate(positions, 4, 16, out));
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(SdpaBitwiseRingMaskGenOutTest, RejectsInvalidShapesAndDtypes) {
  TensorFactory<ScalarType::Long> positions_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  TensorFactory<ScalarType::Float> float_factory;
  Tensor positions = positions_factory.make({2}, {0, 1});
  Tensor wrong_rank = positions_factory.make({1, 2}, {0, 1});
  Tensor wrong_dtype = float_factory.zeros({2});
  Tensor out = byte_factory.zeros({2, 2});
  Tensor wrong_output_rank = byte_factory.zeros({4});
  Tensor wrong_output_rows = byte_factory.zeros({1, 2});
  Tensor wrong_output_width = byte_factory.zeros({2, 1});
  Tensor wrong_output_shape = byte_factory.zeros({1, 4});
  Tensor wrong_output_dtype = float_factory.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, generate(wrong_rank, 4, 16, out));
  ET_EXPECT_KERNEL_FAILURE(context_, generate(wrong_dtype, 4, 16, out));
  for (Tensor* invalid_output :
       {&wrong_output_rank,
        &wrong_output_rows,
        &wrong_output_width,
        &wrong_output_shape}) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, generate(positions, 4, 16, *invalid_output));
  }
  ET_EXPECT_KERNEL_FAILURE(
      context_, generate(positions, 4, 16, wrong_output_dtype));
}
#endif

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
