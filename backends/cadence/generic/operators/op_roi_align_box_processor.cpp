/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_roi_align_box_processor.h>

#include <array>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

using UnpackedVec = std::array<int, 22>;
using PackedVec = std::array<uint8_t, 80>;
using IterVec = std::array<int, 6>;

IterVec computeAddrIncr(const IterVec& shape, const IterVec& strides) {
  auto rank = shape.size();
  auto inc = strides;
  for (int n = 1; n < static_cast<int>(rank); ++n) {
    inc[n] = strides[n] - strides[n - 1] * shape[n - 1] + inc[n - 1];
  }
  return inc;
}

template <int perItemBitWidth = 29>
PackedVec packTuringVals(const UnpackedVec& vals, bool is_signed) {
  PackedVec result{};
  int bitPos = 0; // bit position in output vector
  for (int v : vals) {
    assert(is_signed || v >= 0);
    if (is_signed) {
      assert(
          v >= -(1 << (perItemBitWidth - 1)) &&
          v < (1 << (perItemBitWidth - 1)));
    } else {
      assert(v < (1 << perItemBitWidth));
    }

    if (v < 0) {
      v = (1 << perItemBitWidth) + v;
    }

    // Extract bit by bit and store in the output array
    for (int bit = 0; bit < perItemBitWidth; ++bit) {
      auto outBitIndex = bitPos + bit;
      auto byteIndex = outBitIndex / 8;
      auto bitInByte = outBitIndex % 8;
      // Extract bit from val
      uint8_t bitVal = (v >> bit) & 1;
      // Set bit in output byte
      result[byteIndex] |= (bitVal << bitInByte);
    }
    bitPos += perItemBitWidth;
  }
  assert(bitPos == vals.size() * perItemBitWidth);
  return result;
}

template <int precision_mode, int frac_bits = precision_mode == 0 ? 16 : 8>
constexpr int get_fp_scale() {
  return 1 << frac_bits;
}

template <int precision_mode>
int convert_to_S13(float fp) {
  return int(std::round(fp * get_fp_scale<precision_mode>()));
}

PackedVec convertBoxPosToTuringConfig(
    float topLeftX,
    float topLeftY,
    float bottomRightX,
    float bottomRightY,
    int roiAlignNumBoxes,
    int output_size_h,
    int output_size_w,
    int sampling_ratio,
    bool aligned) {
  constexpr int precisionMode = 0;
  auto dstImgH = output_size_h * sampling_ratio;
  auto dstImgW = output_size_w * sampling_ratio;
  auto dstTileH = dstImgH;
  auto dstTileW = dstImgW;

  float stepX = (bottomRightX - topLeftX) / dstImgW;
  float stepY = (bottomRightY - topLeftY) / dstImgH;

  if (aligned) {
    topLeftX -= 0.5;
    topLeftY -= 0.5;
  }

  auto anchorX = convert_to_S13<precisionMode>(topLeftX + stepX / 2);
  auto anchorY = convert_to_S13<precisionMode>(topLeftY + stepY / 2);

  UnpackedVec vals{};
  vals[0] = anchorX;
  vals[1] = anchorY;

  IterVec shape = {dstTileW, dstTileH, 1, 1, 1, roiAlignNumBoxes};
  auto addrIncrementX = computeAddrIncr(
      shape,
      {convert_to_S13<precisionMode>(stepX),
       0,
       convert_to_S13<precisionMode>(stepX * dstTileW),
       0,
       0,
       0});
  auto addrIncrementY = computeAddrIncr(
      shape,
      {0,
       convert_to_S13<precisionMode>(stepY),
       0,
       convert_to_S13<precisionMode>(stepY * dstTileH),
       0,
       0});

  for (int i = 0; i < 10; ++i) {
    vals[i + 2] = i < addrIncrementX.size()
        ? addrIncrementX[i]
        : addrIncrementX[addrIncrementX.size() - 1];
    vals[i + 12] = i < addrIncrementY.size()
        ? addrIncrementY[i]
        : addrIncrementY[addrIncrementY.size() - 1];
  }

  return packTuringVals(vals, true);
}

} // namespace

Tensor& roi_align_box_processor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& rois,
    int64_t output_size_h,
    int64_t output_size_w,
    int64_t sampling_ratio,
    bool aligned,
    Tensor& out) {
  int K = static_cast<int>(rois.size(0));
  auto roi = rois.const_data_ptr<float>();
  for (int i = 0; i < K; ++i) {
    assert(
        static_cast<int>(roi[i * 5]) == 0 && "Only support 1 image for now.");
    auto x1 = roi[i * 5 + 1];
    auto y1 = roi[i * 5 + 2];
    auto x2 = roi[i * 5 + 3];
    auto y2 = roi[i * 5 + 4];
    auto turing_roi = convertBoxPosToTuringConfig(
        x1,
        y1,
        x2,
        y2,
        static_cast<int>(K),
        static_cast<int>(output_size_h),
        static_cast<int>(output_size_w),
        static_cast<int>(sampling_ratio),
        aligned);
    static_assert(turing_roi.size() == 80);

    auto out_ptr = out.mutable_data_ptr<uint8_t>() + i * turing_roi.size();
    for (auto val : turing_roi) {
      *out_ptr++ = val;
    }
  }
  return out;
}
} // namespace native
} // namespace generic
} // namespace impl
