/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <executorch/backends/cadence/generic/operators/op_roi_align_box_processor.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

using Fields = std::array<int32_t, 22>;

// Each box becomes 80 bytes: 22 signed 29-bit fields, packed LSB first. The
// fields are the anchor (x, y), then 10 x address increments and 10 y address
// increments, all in fixed point with 16 fractional bits (1.0 == 65536). There
// is no reference implementation, so the expected fields are worked out below.
Fields unpack(const uint8_t* bytes) {
  Fields fields{};
  for (int f = 0; f < 22; ++f) {
    uint32_t v = 0;
    for (int bit = 0; bit < 29; ++bit) {
      const int pos = f * 29 + bit;
      v |= static_cast<uint32_t>((bytes[pos / 8] >> (pos % 8)) & 1) << bit;
    }
    fields[f] = static_cast<int32_t>(v << 3) >> 3; // sign-extend 29 bits
  }
  return fields;
}

// For an output of 2x2 with sampling_ratio 1, a box w wide and h tall has steps
// sx = w / 2 and sy = h / 2. The anchor is the top-left corner plus half a
// step. With iteration shape {2, 2, 1, 1, 1, K}, the x increments come out as
// [sx, -sx, sx, -sx, -sx, -sx] and the y increments as [0, sy, -sy, sy, -sy,
// -sy]; the last one repeats to fill 10.
Fields
expected_fields(int32_t anchor_x, int32_t anchor_y, int32_t sx, int32_t sy) {
  Fields f{};
  f[0] = anchor_x;
  f[1] = anchor_y;
  const std::array<int32_t, 6> x{sx, -sx, sx, -sx, -sx, -sx};
  const std::array<int32_t, 6> y{0, sy, -sy, sy, -sy, -sy};
  for (int i = 0; i < 10; ++i) {
    f[2 + i] = x[i < 6 ? i : 5];
    f[12 + i] = y[i < 6 ? i : 5];
  }
  return f;
}

constexpr int32_t kOne = 1 << 16;

class GenericRoiAlignBoxProcessorTest : public OperatorTest {
 protected:
  TensorFactory<ScalarType::Float> tf_float_;
  TensorFactory<ScalarType::Byte> tf_byte_;
};

// Box 0 spans (0, 0)-(4, 4): steps 2 and 2, anchor (1, 1). Box 1 spans (8, 0)-
// (12, 8): steps 2 and 4, anchor (9, 2).
TEST_F(GenericRoiAlignBoxProcessorTest, TwoBoxes) {
  const Tensor rois = tf_float_.make({2, 5}, {0, 0, 0, 4, 4, 0, 8, 0, 12, 8});
  Tensor out = tf_byte_.zeros({2, 80});

  roi_align_box_processor_out(context_, rois, 2, 2, 1, /*aligned=*/false, out);

  const uint8_t* data = out.const_data_ptr<uint8_t>();
  EXPECT_EQ(unpack(data), expected_fields(kOne, kOne, 2 * kOne, 2 * kOne));
  EXPECT_EQ(
      unpack(data + 80),
      expected_fields(9 * kOne, 2 * kOne, 2 * kOne, 4 * kOne));
  // 22 fields take 638 bits; the last 2 bits of each box stay 0.
  EXPECT_EQ(data[79] >> 6, 0);
  EXPECT_EQ(data[159] >> 6, 0);
}

// aligned moves the corner by half a pixel, so the anchor is (0.5, 0.5); the
// steps do not change.
TEST_F(GenericRoiAlignBoxProcessorTest, Aligned) {
  const Tensor rois = tf_float_.make({1, 5}, {0, 0, 0, 4, 4});
  Tensor out = tf_byte_.zeros({1, 80});

  roi_align_box_processor_out(context_, rois, 2, 2, 1, /*aligned=*/true, out);

  EXPECT_EQ(
      unpack(out.const_data_ptr<uint8_t>()),
      expected_fields(kOne / 2, kOne / 2, 2 * kOne, 2 * kOne));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
