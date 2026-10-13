/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/quantized_linear.h>
#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/backends/cadence/hifi/operators/operators.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cmath>
#include <cstring>

// Row-packed sub-byte fully-connected on HiFi.
//
// The weight arrives as [out_dim, packed_row_bytes] int8, where the bytes are
// a container rather than values, so `in_dim` and `weight_bits` are passed
// explicitly. Byte order follows the AoT packer in
// executorch/backends/cadence/aot/weight_packing.py (a planar 4+2 layout for
// 6-bit, two nibbles per byte for 4-bit), values in offset binary.
//
// Shape of the kernel:
//
//   nnlib has no sub-byte support on HiFi4 - xa_nn_matmul_asym4sxasym8s_asym8s
//   and friends are `return -1` stubs, and 6-bit does not exist at all - so the
//   unpacking is ours. But nnlib *does* have a per-channel matmul,
//   xa_nn_matmul_per_chan_sym8sxasym8s_asym8s, which is a fully vectorized
//   int8 GEMM with exactly the per-channel requantize this op needs.
//
//   So we unpack a tile of rows into scratch and hand the tile to nnlib. That
//   keeps the tuned MAC and requantize instead of reimplementing them, and
//   bounds scratch at a few KB rather than unpacking the whole
//   matrix. Weights stream from DRAM packed, which is where the bandwidth win
//   of sub-byte weights actually shows up: for the wakeword shapes this is a
//   GEMV, so it is memory bound on weight traffic and 6-bit moves 25% fewer
//   bytes than 8-bit.
//
// There is no fatal path here. Which kernel serves this operator is decided at
// build time by operator_fallback.bzl, so there is no generic kernel linked to
// fall back to at runtime; aborting would take the model down. Anything the
// nnlib fast path cannot express - a non-int8 dtype, a non-symmetric weight, a
// failed scratch allocation - falls to the scalar path below instead.
//
// Requantize convention: this follows nnlib, i.e. a positive out_multiplier
// scales positively. That matches the neighbouring 8-bit HiFi kernels
// (op_quantized_fully_connected_out.cpp hands out_multiplier to nnlib
// unchanged) and therefore the deployed per-tensor behaviour. Note it does
// *not* bit-match the generic reference kernel, which applies
// `-out_multiplier`; that discrepancy is pre-existing and independent of
// packing - see the note in quantized_linear_packed.h.

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

namespace {

// Rows unpacked per tile. nnlib's int8 matmul has an 8-rows-at-a-time inner
// loop, so the tile has to be a multiple of 8 to keep it on that path, and 8
// is the floor.
constexpr int64_t kMinRowTile = 8;
// Ceiling on the scratch tile, in bytes. Everything above the floor is bought
// with scratch, and the tile is written once and read once straight after, so
// it wants to stay resident. 8KB is where the curve flattens: measured on the
// 4-bit wakeword stage 2 (RT600, cycles in ops), 4KB gave 1,058,885, 8KB
// 1,035,586 and 16KB 1,029,294 - the last doubling buys 0.6%, which is not
// worth 8KB more of a 64KB DTCM.
constexpr int64_t kTileBudgetBytes = 8192;
constexpr int64_t kMaxRowTile = 128;

// Rows per tile for a given in_dim, spending the byte budget rather than
// fixing the row count.
//
// The fixed cost of a tile - the nnlib call, the bias fold, the decode
// prologue - is paid per tile, not per value, so a matrix with a short row
// pays it far more often for the same total work. The two wakeword shapes are
// identical in bytes (96x480 and 480x96) and the 480-row one measured 1.7x the
// cycles; at a fixed 8 rows it makes 60 nnlib calls against the other's 12.
// Budgeting by bytes gives both of them a tile of about the same size, and
// took the 4-bit model from 1,124,634 cycles to 1,035,586.
inline int64_t rows_per_tile(const int64_t in_dim) {
  const int64_t budgeted = (kTileBudgetBytes / in_dim) & ~(kMinRowTile - 1);
  if (budgeted < kMinRowTile) {
    return kMinRowTile;
  }
  return budgeted > kMaxRowTile ? kMaxRowTile : budgeted;
}

// nnlib's HiFi1S per-channel matmul requantizes rows in pairs and applies the
// second row's shift to both, so a tile whose paired rows have different
// shifts has to be fed one row at a time.
inline int64_t nnlib_rows_per_call(
    const int32_t* __restrict__ shift,
    const int64_t rows) {
#if defined(XCHAL_HAVE_HIFI1S) && XCHAL_HAVE_HIFI1S
  for (int64_t t = 0; t + 1 < rows; t += 2) {
    if (shift[t] != shift[t + 1]) {
      return 1;
    }
  }
#else
  (void)shift;
#endif
  return rows;
}

// Values per packed group, and bytes those values occupy.
// Scalar planar 4+2 accessor. Mirrors decode_planar_6bit() in the generic
// kernel's quantized_linear_packed.h; the two are cross-checked on the host.
// Used wherever the vectorized path does not apply - a row whose in_dim is not
// a multiple of 32, or an unaligned row.
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

// Scalar split 4-bit accessor. Mirrors decode_split_4bit() in the generic
// kernel's quantized_linear_packed.h.
inline __attribute__((always_inline)) int32_t decode_split_4bit(
    const uint8_t* __restrict__ row,
    const int64_t k,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  const uint8_t b = row[k < half ? k : k - half];
  const uint8_t nibble = k < half ? (b & 0x0F) : ((b >> 4) & 0x0F);
  return static_cast<int32_t>(nibble) - 8;
}

// One decoded weight in natural k order, for any supported width. 4- and
// 6-bit are split layouts with no per-group form; 8-bit is stored as-is.
inline __attribute__((always_inline)) int32_t decode_weight(
    const uint8_t* __restrict__ row,
    const int64_t k,
    const int64_t in_dim,
    const int64_t bits) {
  if (bits == 6) {
    return decode_planar_6bit(row, k, in_dim);
  }
  if (bits == 4) {
    return decode_split_4bit(row, k, in_dim);
  }
  return static_cast<int32_t>(static_cast<int8_t>(row[k]));
}

// ---------------------------------------------------------------------------
// Permuted 6-bit decode.
//
// Writing the weights in natural k order needs a cross-lane shuffle: the packer
// emits 4 values per 3 bytes, so input stride 3 does not line up with output
// stride 4. But a dot product is invariant under any permutation of k applied
// to BOTH operands, and this is a GEMV - the activation vector is reused across
// every one of the out_dim rows, so permuting it costs O(in_dim) once and is
// amortised over 96..480 rows.
//
// Under the permutation "all the low-6-bit values in byte order, then all the
// split values", 3 of every 4 weights become simply (byte & 0x3F) at the same
// index - contiguous in and contiguous out, no shuffle. Only the remaining
// quarter needs the cross-byte gather.
//
// The offset binary bias folds away too. With u = byte & 0x3F the weight is
// u - 32, and sum_k (x_k - zp)(u_k - 32) = sum_k (x_k - zp) u_k - 32 * S with
// S = sum_k (x_k - zp). So S is folded into the bias and the decode never
// subtracts: the easy three quarters are one AND and nothing else.

// Decode one planar 4+2 row into `dst`, leaving values in [0, 63]; the -32
// offset binary bias is carried by the bias adjustment instead.
//
// Because the packer pairs value k with value k + in_dim/2 (rather than k with
// k+1), masking the low nibbles of a word yields four consecutive values. So
// the decode writes four contiguous runs in natural k order and no activation
// permutation is needed - the generic kernel and this one see the same order.
//
// 32 values per iteration: 3 loads, 17 ALU, 4 stores, all elementwise on
// 2-lane ae_int32x2. Nothing straddles a byte, which is the whole reason for
// the planar layout.
// Vector form, for in_dim a multiple of 32 (so quarter is a multiple of 8 and
// all four runs land on 8-byte boundaries).
//
// There is deliberately no unaligned AE twin. One existed and was wrong: the
// AE_ZALIGN64/AE_SA32X2_IP/AE_SA64POS_FP idiom writes whole 8-byte containers
// and zero-fills the bytes ahead of an unaligned start, so with four store
// streams at dst+0/+Q/+2Q/+3Q each stream's first store scribbled into the
// previous run. It cost ~87k cycles less than the scalar form and raised MSE
// from 1.61e-04 to 1.28e-03 while still passing the 5e-3 threshold. Shapes
// that cannot use the aligned form take unpack_row_planar_6bit_scalar().
inline void unpack_row_planar_6bit_aligned(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  const int64_t quarter = in_dim >> 2;
  const ae_int32x2 m0F = AE_MOVDA32(0x0F0F0F0F);
  const ae_int32x2 m30 = AE_MOVDA32(0x30303030);
  const ae_int32x2* __restrict__ pb =
      reinterpret_cast<const ae_int32x2*>(src + half);
  ae_int32x2 a0, a1, b0;

  // The cursors are rebuilt from k on purpose. Hoisting them and letting the
  // _IP post-increment carry them across iterations measured 1.1% slower: the
  // compiler already strength-reduces these, and the mutable carried pointers
  // only constrain its scheduling.
  for (int64_t k = 0; k < quarter; k += 8) {
    const ae_int32x2* __restrict__ pa0 =
        reinterpret_cast<const ae_int32x2*>(src + k);
    const ae_int32x2* __restrict__ pa1 =
        reinterpret_cast<const ae_int32x2*>(src + k + quarter);
    AE_L32X2_IP(a0, pa0, 8);
    AE_L32X2_IP(a1, pa1, 8);
    AE_L32X2_IP(b0, pb, 8);

    ae_int32x2* __restrict__ d0 = reinterpret_cast<ae_int32x2*>(dst + k);
    ae_int32x2* __restrict__ d1 =
        reinterpret_cast<ae_int32x2*>(dst + k + quarter);
    ae_int32x2* __restrict__ d2 =
        reinterpret_cast<ae_int32x2*>(dst + k + 2 * quarter);
    ae_int32x2* __restrict__ d3 =
        reinterpret_cast<ae_int32x2*>(dst + k + 3 * quarter);
    AE_S32X2_IP(
        AE_OR32(AE_AND32(a0, m0F), AE_AND32(AE_SLLI32(b0, 4), m30)), d0, 8);
    AE_S32X2_IP(
        AE_OR32(AE_AND32(a1, m0F), AE_AND32(AE_SLLI32(b0, 2), m30)), d1, 8);
    AE_S32X2_IP(
        AE_OR32(AE_AND32(AE_SRLI32(a0, 4), m0F), AE_AND32(b0, m30)), d2, 8);
    AE_S32X2_IP(
        AE_OR32(
            AE_AND32(AE_SRLI32(a1, 4), m0F), AE_AND32(AE_SRLI32(b0, 2), m30)),
        d3,
        8);
  }
}

// Scalar planar decode. Deliberately the plainest possible statement of the
// layout, used to isolate the vector paths. Note it does NOT apply the -32
// offset-binary bias: on the planar path that is folded into tile_bias via
// offset_fold, unlike unpack_row() which does subtract it.
inline void unpack_row_planar_6bit_scalar(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  const int64_t quarter = in_dim >> 2;
  for (int64_t k = 0; k < quarter; ++k) {
    const uint8_t a = src[k];
    const uint8_t c = src[k + quarter];
    const uint8_t b = src[half + k];
    dst[k] = static_cast<int8_t>((a & 0x0F) | ((b << 4) & 0x30));
    dst[k + quarter] = static_cast<int8_t>((c & 0x0F) | ((b << 2) & 0x30));
    dst[k + 2 * quarter] = static_cast<int8_t>(((a >> 4) & 0x0F) | (b & 0x30));
    dst[k + 3 * quarter] =
        static_cast<int8_t>(((c >> 4) & 0x0F) | ((b >> 2) & 0x30));
  }
}

inline void unpack_row_planar_6bit(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim,
    const bool aligned) {
  if (aligned) {
    unpack_row_planar_6bit_aligned(dst, src, in_dim);
  } else {
    unpack_row_planar_6bit_scalar(dst, src, in_dim);
  }
}

// 4-bit split. Byte k holds value k in the low nibble and value k + in_dim/2
// in the high nibble, so one masked word yields 8 consecutive values and the
// shifted-and-masked word yields the 8 that live half a row away. Nothing
// straddles a byte and nothing needs de-interleaving.
//
// This is the cheapest of the three widths by some margin: 1 load, 3 ALU and
// 2 stores per 16 values, against 3 loads, 17 ALU and 4 stores per 32 values
// for 6-bit - and it moves 0.5 bytes per value instead of 0.75. 4 divides 8,
// so unlike 6-bit there is no leftover-bit problem to pay for at all.
//
// Needs half to be a multiple of 16, i.e. in_dim a multiple of 32 - the same
// gate as the 6-bit vector decode.
//
// Unrolled 2x deliberately. The natural 8-wide form does 16 values per
// iteration against the 6-bit decode's 32, so for a given row it runs twice as
// many iterations; with a body this short the loop overhead dominated and the
// per-value instruction win was lost (measured: 4-bit came out ~12% slower
// than 6-bit that way, on strictly less traffic). At 32 values per iteration
// it is 2 loads, 6 ALU and 4 stores against 6-bit's 3 loads, 17 ALU and 4
// stores, and moves 0.5 bytes per value instead of 0.75.
//
// Takes the whole row tile rather than one row, because per-row cost is what
// dominates for a tall matrix. Fitting the isolated per-layer measurements to
// `a * rows + b * values` over the two wakeword shapes (96x480 and 480x96,
// identical in bytes) gives a = 120 cycles per row against b = 0.91 cycles per
// value: a 480-row layer spends more than a third of its time on a fixed cost
// paid once per row, over an inner loop that runs 3 times. Hoisting the row
// loop in here pays the constant setup - the 0x0F mask, the loop bounds - once
// per tile instead of once per row, and measured 1,211,677 -> 1,124,634 cycles
// on the 4-bit wakeword stage 2, bit-identical output.
//
// The three cursors are post-incremented and live across the inner loop. The
// 6-bit decode rebuilds its cursors from k each iteration, and measured faster
// that way, but it has four store streams whose addresses the compiler can
// strength-reduce independently; here there are three, they advance in
// lockstep, and _IP folds the update into the load or store, so the inner loop
// carries no address arithmetic at all.
inline void unpack_tile_split_4bit_aligned(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim,
    const int64_t row_bytes,
    const int64_t rows) {
  const int64_t half = in_dim >> 1;
  const ae_int32x2 m0F = AE_MOVDA32(0x0F0F0F0F);

  for (int64_t t = 0; t < rows; ++t) {
    const ae_int32x2* __restrict__ p =
        reinterpret_cast<const ae_int32x2*>(src + t * row_bytes);
    ae_int32x2* __restrict__ dlo =
        reinterpret_cast<ae_int32x2*>(dst + t * in_dim);
    ae_int32x2* __restrict__ dhi = dlo + (half >> 3);

    // 32 values per iteration. This width is a measured local optimum, not a
    // guess: 16 values per iteration cost 276k cycles more (the loop ran twice
    // as many times for a body this short), and 64 values per iteration cost
    // 2.6% more again on register pressure.
    for (int64_t k = 0; k < half; k += 16) {
      ae_int32x2 a0, a1;
      AE_L32X2_IP(a0, p, 8);
      AE_L32X2_IP(a1, p, 8);
      AE_S32X2_IP(AE_AND32(a0, m0F), dlo, 8);
      AE_S32X2_IP(AE_AND32(a1, m0F), dlo, 8);
      AE_S32X2_IP(AE_AND32(AE_SRLI32(a0, 4), m0F), dhi, 8);
      AE_S32X2_IP(AE_AND32(AE_SRLI32(a1, 4), m0F), dhi, 8);
    }
  }
}

inline void unpack_row_split_4bit_scalar(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim) {
  const int64_t half = in_dim >> 1;
  for (int64_t k = 0; k < half; ++k) {
    const uint8_t b = src[k];
    dst[k] = static_cast<int8_t>(b & 0x0F);
    dst[k + half] = static_cast<int8_t>((b >> 4) & 0x0F);
  }
}

inline void unpack_tile_split_4bit(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim,
    const int64_t row_bytes,
    const int64_t rows,
    const bool aligned) {
  if (aligned) {
    unpack_tile_split_4bit_aligned(dst, src, in_dim, row_bytes, rows);
    return;
  }
  for (int64_t t = 0; t < rows; ++t) {
    unpack_row_split_4bit_scalar(dst + t * in_dim, src + t * row_bytes, in_dim);
  }
}

// Decode one packed row of `in_dim` values in natural k order.
//
// This walks the row as a byte stream, which is the layout the packer emits,
// so it is sequential rather than a gather. packed_row_bytes() rejects an
// in_dim that is not a multiple of the group size, so there is no tail.
inline void unpack_row(
    int8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t in_dim,
    const int64_t weight_bits) {
  if (weight_bits == 6) {
    // Walk the four output runs directly. decode_planar_6bit() is the random
    // access form and costs a divide and a modulo per element; iterating the
    // structure instead makes both index terms loop-invariant.
    const int64_t half = in_dim >> 1;
    const int64_t quarter = in_dim >> 2;
    for (int64_t q = 0; q < 4; ++q) {
      const int64_t out_base = q * quarter;
      const int64_t a_base = (q < 2) ? out_base : out_base - half;
      const int shift = static_cast<int>(2 * q);
      for (int64_t k = 0; k < quarter; ++k) {
        const uint8_t a = src[a_base + k];
        const uint8_t nibble = (q < 2) ? (a & 0x0F) : ((a >> 4) & 0x0F);
        const uint8_t field = (src[half + k] >> shift) & 0x03;
        dst[out_base + k] = static_cast<int8_t>(
            static_cast<int32_t>(nibble | (field << 4)) - 32);
      }
    }
    return;
  }
  if (weight_bits == 4) {
    // Subtracts the offset binary bias, unlike unpack_row_split_4bit_scalar():
    // this path runs when the tile_bias allocation failed, so offset_fold is 0
    // and nothing downstream will apply it.
    const int64_t half = in_dim >> 1;
    for (int64_t k = 0; k < half; ++k) {
      const uint8_t b = src[k];
      dst[k] = static_cast<int8_t>(static_cast<int32_t>(b & 0x0F) - 8);
      dst[k + half] =
          static_cast<int8_t>(static_cast<int32_t>((b >> 4) & 0x0F) - 8);
    }
    return;
  }
  std::memcpy(dst, src, static_cast<size_t>(in_dim));
}

// Scalar path. Slower, but computes the same thing for any dtype, any weight
// zero point, and with no scratch, so it can serve as the no-abort fallback.
template <typename T>
void packed_fully_connected_scalar(
    const T* __restrict__ in_data,
    const uint8_t* __restrict__ w_data,
    const int32_t* __restrict__ bias_data,
    const int32_t* __restrict__ mult_data,
    const int32_t* __restrict__ shift_data,
    const int64_t qparam_stride,
    const int64_t leading_dims,
    const int64_t out_dim,
    const int64_t in_dim,
    const int64_t row_bytes,
    const int64_t weight_bits,
    const int32_t in_zero_point,
    const int32_t weight_zp,
    const int32_t out_zero_point,
    T* __restrict__ out_data) {
  for (int64_t i = 0; i < leading_dims; ++i) {
    for (int64_t j = 0; j < out_dim; ++j) {
      const uint8_t* __restrict__ row = w_data + j * row_bytes;
      int32_t sum = bias_data[j];
      for (int64_t k = 0; k < in_dim; ++k) {
        sum += (static_cast<int32_t>(in_data[i * in_dim + k]) - in_zero_point) *
            (decode_weight(row, k, in_dim, weight_bits) - weight_zp);
      }
      // nnlib's convention, matching the fast path: a positive multiplier
      // scales positively.
      const float requant_scale =
          static_cast<float>(mult_data[j * qparam_stride]) / 2147483648.0f *
          std::pow(2.0f, shift_data[j * qparam_stride]);
      out_data[i * out_dim + j] =
          kernels::quantize<T>(sum, requant_scale, out_zero_point);
    }
  }
}

} // namespace

void quantized_fully_connected_packed_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_dim,
    int64_t weight_bits,
    int64_t in_zero_point,
    const optional<Tensor>& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  const int64_t out_dim = weight.size(0);
  const int64_t row_bytes = weight.size(1);
  const int64_t leading_dims =
      ::executorch::runtime::getLeadingDims(in, in.dim() - 1);

  const int32_t weight_zp =
      ::impl::generic::quantized::resolve_weight_zero_point(weight_zero_point);
  const int32_t* __restrict__ mult_data =
      out_multiplier.const_data_ptr<int32_t>();
  const int32_t* __restrict__ shift_data = out_shift.const_data_ptr<int32_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  const uint8_t* __restrict__ w_data =
      reinterpret_cast<const uint8_t*>(weight.const_data_ptr<int8_t>());

  // A per-tensor caller lifts its scalar qparams to length-1 tensors, so the
  // channel index has to collapse to 0 for them.
  const bool qparams_are_scalar = out_multiplier.numel() == 1;
  const int64_t qparam_stride = qparams_are_scalar ? 0 : 1;

  // nnlib's per-channel matmul is sym8s on int8: it has no weight zero-point
  // operand at all, and handles int8 only. Per-channel weights are quantized
  // symmetrically so this normally holds.
  const bool fast_path = out.scalar_type() == ScalarType::Char &&
      in.scalar_type() == ScalarType::Char && weight_zp == 0;

  const int64_t row_tile = rows_per_tile(in_dim);
  int8_t* __restrict__ tile = nullptr;
  int32_t* __restrict__ tile_mult = nullptr;
  int32_t* __restrict__ tile_shift = nullptr;
  if (fast_path) {
    tile = reinterpret_cast<int8_t*>(
        kernels::allocate_temp_memory(ctx, row_tile * in_dim));
    if (qparams_are_scalar) {
      tile_mult = reinterpret_cast<int32_t*>(
          kernels::allocate_temp_memory(ctx, row_tile * sizeof(int32_t)));
      tile_shift = reinterpret_cast<int32_t*>(
          kernels::allocate_temp_memory(ctx, row_tile * sizeof(int32_t)));
    }
  }
  // A failed scratch allocation is not fatal: drop to the scalar path, which
  // needs none.
  const bool have_scratch = tile != nullptr &&
      (!qparams_are_scalar || (tile_mult != nullptr && tile_shift != nullptr));

  if (!fast_path || !have_scratch) {
#define typed_packed_fc_scalar(ctype, dtype)  \
  case ScalarType::dtype: {                   \
    packed_fully_connected_scalar<ctype>(     \
        in.const_data_ptr<ctype>(),           \
        w_data,                               \
        bias_data,                            \
        mult_data,                            \
        shift_data,                           \
        qparam_stride,                        \
        leading_dims,                         \
        out_dim,                              \
        in_dim,                               \
        row_bytes,                            \
        weight_bits,                          \
        static_cast<int32_t>(in_zero_point),  \
        weight_zp,                            \
        static_cast<int32_t>(out_zero_point), \
        out.mutable_data_ptr<ctype>());       \
    break;                                    \
  }
    switch (out.scalar_type()) {
      ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_packed_fc_scalar);
      default:
        ET_DCHECK_MSG(
            false,
            "Unhandled dtype %s",
            torch::executor::toString(out.scalar_type()));
    }
#undef typed_packed_fc_scalar
    return;
  }

  if (qparams_are_scalar) {
    for (int64_t t = 0; t < row_tile; ++t) {
      tile_mult[t] = mult_data[0];
      tile_shift[t] = shift_data[0];
    }
  }

  const int8_t* __restrict__ in_data = in.const_data_ptr<int8_t>();
  int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

  // 4- and 6-bit are both split layouts: the decode emits values in natural k
  // order with the offset-binary bias left folded into the bias below, so both
  // take the same shape here. 8-bit is stored as-is.
  int32_t* __restrict__ tile_bias = nullptr;
  if (weight_bits == 6 || weight_bits == 4) {
    tile_bias = reinterpret_cast<int32_t*>(
        kernels::allocate_temp_memory(ctx, row_tile * sizeof(int32_t)));
  }
  // The vector decodes both need a run length that is a multiple of 8 values
  // per store stream, which for either width lands on in_dim being a multiple
  // of 32: 6-bit runs are in_dim/4 and 4-bit is unrolled to 2 x in_dim/2. The
  // wakeword stem (in_dim = 200) qualifies for neither and takes the scalar
  // decode.
  const bool row_aligned = ((in_dim & 31) == 0) && ((row_bytes & 7) == 0) &&
      ((reinterpret_cast<uintptr_t>(w_data) & 7) == 0);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // The word gather's shift constants assume little-endian byte order.
  constexpr bool kByteOrderOk = false;
#else
  constexpr bool kByteOrderOk = true;
#endif
  const bool split_layout = tile_bias != nullptr && kByteOrderOk;
  // Offset binary: the decode leaves u in [0, 2^bits - 1] and the real weight
  // is u - 2^(bits-1).
  const int32_t weight_offset = 1 << (weight_bits - 1);

  for (int64_t i = 0; i < leading_dims; ++i) {
    const int8_t* __restrict__ vec = in_data + i * in_dim;
    int32_t offset_fold = 0;
    if (split_layout) {
      // sum (x-zp)(u-B) = sum (x-zp)u - B*S with S = sum(x - zp), so S goes
      // into the bias and the decode never subtracts.
      int32_t s = 0;
      for (int64_t k = 0; k < in_dim; ++k) {
        s += static_cast<int32_t>(vec[k]);
      }
      offset_fold = -weight_offset *
          (s -
           static_cast<int32_t>(in_dim) * static_cast<int32_t>(in_zero_point));
    }

    for (int64_t r0 = 0; r0 < out_dim; r0 += row_tile) {
      const int64_t rows =
          (out_dim - r0) < row_tile ? (out_dim - r0) : row_tile;

      if (split_layout && weight_bits == 4) {
        // Whole tile in one call: the fixed per-row cost is what dominates a
        // tall matrix, so it is paid once per tile here instead of once per
        // row.
        unpack_tile_split_4bit(
            tile,
            w_data + r0 * row_bytes,
            in_dim,
            row_bytes,
            rows,
            row_aligned);
      } else {
        for (int64_t t = 0; t < rows; ++t) {
          const uint8_t* __restrict__ row = w_data + (r0 + t) * row_bytes;
          if (!split_layout) {
            unpack_row(tile + t * in_dim, row, in_dim, weight_bits);
          } else {
            unpack_row_planar_6bit(tile + t * in_dim, row, in_dim, row_aligned);
          }
        }
      }
      if (split_layout) {
        for (int64_t t = 0; t < rows; ++t) {
          tile_bias[t] = bias_data[r0 + t] + offset_fold;
        }
      }

      const int32_t* __restrict__ call_bias =
          split_layout ? tile_bias : bias_data + r0;
      const int32_t* __restrict__ call_mult =
          qparams_are_scalar ? tile_mult : mult_data + r0;
      const int32_t* __restrict__ call_shift =
          qparams_are_scalar ? tile_shift : shift_data + r0;
      const int64_t rows_per_call = nnlib_rows_per_call(call_shift, rows);
      for (int64_t t = 0; t < rows; t += rows_per_call) {
        const int32_t ret = xa_nn_matmul_per_chan_sym8sxasym8s_asym8s(
            out_data + i * out_dim + r0 + t,
            tile + t * in_dim, // p_mat1: [rows, in_dim] unpacked weights
            vec, // p_vec1
            call_bias + t,
            rows_per_call, // rows
            in_dim, // cols1
            in_dim, // row_stride1
            1, // vec_count: one vector at a time
            in_dim, // vec_offset
            out_dim, // out_offset
            1, // out_stride
            -in_zero_point, // nnlib takes the negated zero point
            call_mult + t,
            call_shift + t,
            out_zero_point);
        ET_DCHECK_MSG(ret == 0, "HiFi quantized_fully_connected_packed failed");
      }
    }
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
