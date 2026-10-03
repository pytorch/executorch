/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>
#include <cmath>
#include <cstdint>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

// Reference kernel for row-packed sub-byte weights.
//
// The weight arrives as an int8 tensor of shape [out_dim, packed_row_bytes]:
// the bytes are a container, not values. `in_dim` and `weight_bits` come in as
// explicit arguments because they can no longer be read off the shape.
//
// Byte order matches the AoT packer in
// executorch/backends/cadence/aot/weight_packing.py, which in turn follows
// the planar 4+2 layout. Signed values are offset binary (+2^(bits-1)).
//
// This is deliberately plain: no vectorization, no tiling. It exists to be
// obviously correct and to serve as the oracle that an optimized HiFi kernel
// gets validated against.

namespace impl::generic::quantized {

// Decode one group of packed values into `dst`, which must have room for the
// group. Returns the number of bytes consumed.
//
// The loop below walks k in order for a fixed output channel, and that is the
// axis the packer packed along, so decoding is a sequential stream rather than
// an indexed gather. That is the whole point of packing per row.
// 4- and 6-bit are both split layouts (see decode_split_4bit and
// decode_planar_6bit) and have no per-group form, so only 8-bit is handled
// here.
inline __attribute__((always_inline)) int decode_group(
    int32_t* __restrict__ dst,
    const uint8_t* __restrict__ p,
    const int64_t bits) {
  (void)bits;
  // 8-bit: stored as-is, already signed
  dst[0] = static_cast<int32_t>(static_cast<int8_t>(p[0]));
  return 1;
}

// 4-bit split. Byte k holds value k in its low nibble and value k + in_dim/2
// in its high nibble - the same pairing as plane A of the 6-bit layout below,
// and for the same reason: masking the low nibbles of a whole word yields
// consecutive values, so the HiFi decode needs one AND and one shift per word
// and no de-interleave. The obvious k-with-k+1 packing would instead produce
// every other value per mask.
inline __attribute__((always_inline)) int32_t decode_split_4bit(
    const uint8_t* __restrict__ row,
    const int64_t k,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  const uint8_t b = row[k < half ? k : k - half];
  const uint8_t nibble = k < half ? (b & 0x0F) : ((b >> 4) & 0x0F);
  return static_cast<int32_t>(nibble) - 8;
}

// 6-bit planar 4+2. The row is two bit-planes rather than groups of four
// values in three bytes:
//
//   plane A, in_dim/2 bytes: byte k holds the low 4 bits of value k in its low
//   nibble and of value k + in_dim/2 in its high nibble
//   plane B, in_dim/4 bytes: byte m holds the high 2 bits of values m, m+Q,
//   m+2Q and m+3Q at bit offsets 0, 2, 4 and 6, where Q = in_dim/4
//
// Same 0.75 bytes per value as the group form, but no value straddles a byte,
// which is what lets the HiFi kernel decode with whole-word masks instead of a
// stride-3 gather. Pairing k with k+in_dim/2 (rather than k with k+1) is what
// keeps the decode in natural k order.
inline __attribute__((always_inline)) int32_t decode_planar_6bit(
    const uint8_t* __restrict__ row,
    const int64_t k,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  const int64_t quarter = in_dim >> 2;
  const uint8_t a = row[k < half ? k : k - half];
  const uint8_t nibble = k < half ? (a & 0x0F) : ((a >> 4) & 0x0F);
  const uint8_t b = row[half + (k % quarter)];
  const uint8_t field = (b >> (2 * (k / quarter))) & 0x03;
  return static_cast<int32_t>(nibble | (field << 4)) - 32;
}

inline __attribute__((always_inline)) int64_t group_values(const int64_t bits) {
  return bits == 4 ? 2 : (bits == 6 ? 4 : 1);
}

// Per-channel requantization, packed weights. Mirrors
// quantized_linear_per_channel_ but decodes the weight row as it walks it.
template <typename T>
inline __attribute__((always_inline)) void quantized_linear_packed_per_channel_(
    const ::executorch::aten::Tensor& src,
    const ::executorch::aten::Tensor& packed_weight,
    const ::executorch::aten::Tensor& bias,
    const int64_t in_dim,
    const int64_t weight_bits,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    const int64_t out_zero_point,
    ::executorch::aten::Tensor& out) {
  const int64_t leading_dims =
      ::executorch::runtime::getLeadingDims(src, src.dim() - 1);
  // out_dim still rides along as shape[0]: that is why we pack per row.
  const int64_t out_dim = packed_weight.size(0);
  const int64_t row_bytes = packed_weight.size(1);

  const int64_t gv = group_values(weight_bits);
  const int32_t src_zp = static_cast<int32_t>(src_zero_point);
  const int32_t weight_zp = static_cast<int32_t>(weight_zero_point);
  const int32_t out_zp = static_cast<int32_t>(out_zero_point);

  const T* __restrict__ in_data = src.const_data_ptr<T>();
  const uint8_t* __restrict__ w_data =
      reinterpret_cast<const uint8_t*>(packed_weight.const_data_ptr<int8_t>());
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();
  const int32_t* __restrict__ out_multiplier_data =
      out_multiplier.const_data_ptr<int32_t>();
  const int32_t* __restrict__ out_shift_data =
      out_shift.const_data_ptr<int32_t>();
  // A per-tensor source lifts its scalar qparams to length-1 tensors, so the
  // channel index has to collapse to 0 for them. quantized_linear.h splits
  // this into a separate per-tensor kernel; striding keeps one loop here and
  // computes the identical expression, which is what bit-exactness needs.
  const int64_t qparam_stride = out_multiplier.numel() == 1 ? 0 : 1;
  ET_CHECK_MSG(
      out_multiplier.numel() == 1 || out_multiplier.numel() == out_dim,
      "out_multiplier must have 1 or out_dim (%" PRId64
      ") elements, got %" PRId64,
      out_dim,
      static_cast<int64_t>(out_multiplier.numel()));
  ET_CHECK_MSG(
      out_shift.numel() == out_multiplier.numel(),
      "out_shift must match out_multiplier in size (%" PRId64 "), got %" PRId64,
      static_cast<int64_t>(out_multiplier.numel()),
      static_cast<int64_t>(out_shift.numel()));

  // Enough room for the widest group we support.
  int32_t decoded[8];

  for (int64_t i = 0; i < leading_dims; ++i) {
    for (int64_t j = 0; j < out_dim; ++j) {
      const uint8_t* __restrict__ row = w_data + j * row_bytes;
      int32_t sum = bias_data[j];
      if (weight_bits == 6) {
        for (int64_t k = 0; k < in_dim; ++k) {
          const int32_t x =
              static_cast<int32_t>(in_data[i * in_dim + k]) - src_zp;
          const int32_t w = decode_planar_6bit(row, k, in_dim) - weight_zp;
          sum += x * w;
        }
      } else if (weight_bits == 4) {
        for (int64_t k = 0; k < in_dim; ++k) {
          const int32_t x =
              static_cast<int32_t>(in_data[i * in_dim + k]) - src_zp;
          const int32_t w = decode_split_4bit(row, k, in_dim) - weight_zp;
          sum += x * w;
        }
      } else {
        int64_t k = 0;
        int64_t byte_off = 0;
        while (k < in_dim) {
          byte_off += decode_group(decoded, row + byte_off, weight_bits);
          for (int64_t t = 0; t < gv && k < in_dim; ++t, ++k) {
            const int32_t x =
                static_cast<int32_t>(in_data[i * in_dim + k]) - src_zp;
            const int32_t w = decoded[t] - weight_zp;
            sum += x * w;
          }
        }
      }
      // Matches quantized_linear.h exactly, including the (1 << 31). The
      // packed and dense kernels have to agree bit for bit, so this expression
      // is deliberately not "cleaned up".
      const float requant_scale =
          -out_multiplier_data[j * qparam_stride] * 1.0 / (1 << 31) *
          std::pow(2, out_shift_data[j * qparam_stride]);
      out_data[i * out_dim + j] = ::impl::generic::kernels::quantize<T>(
          sum, requant_scale, out_zp);
    }
  }
}

} // namespace impl::generic::quantized
