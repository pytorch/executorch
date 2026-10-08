/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace impl {
namespace HiFi {
namespace native {

void quantized_fully_connected_packed_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_dim,
    int64_t weight_bits,
    int64_t in_zero_point,
    const std::optional<::executorch::aten::Tensor>& weight_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    int64_t out_zero_point,
    const std::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::testing::TensorFactory;

constexpr int32_t kInZeroPoint = -5;
constexpr int32_t kOutZeroPoint = 3;
// With a multiplier of 2^30 (0.5) the requantize scale is exactly 2^(shift-1),
// so shifts 1 and 2 scale by exactly 1 and 2 and no rounding convention can
// change the result.
constexpr int32_t kHalfMultiplier = 1 << 30;

// Mirror of the AoT packer in
// executorch/backends/cadence/aot/weight_packing.py, written independently of
// the kernel's decode.
std::vector<int8_t> pack(
    const std::vector<int8_t>& values,
    const int64_t in_dim,
    const int64_t bits) {
  const int64_t half = in_dim / 2;
  const int64_t quarter = in_dim / 4;
  const int32_t offset = 1 << (bits - 1);
  std::vector<int8_t> out;
  for (size_t base = 0; base < values.size(); base += in_dim) {
    auto u = [&](int64_t k) {
      return static_cast<uint8_t>(values[base + k] + offset);
    };
    for (int64_t k = 0; k < half; ++k) {
      out.push_back(
          static_cast<int8_t>((u(k) & 0x0F) | ((u(k + half) & 0x0F) << 4)));
    }
    if (bits == 6) {
      for (int64_t m = 0; m < quarter; ++m) {
        uint8_t b = 0;
        for (int64_t q = 0; q < 4; ++q) {
          b |= static_cast<uint8_t>(
              ((u(m + q * quarter) >> 4) & 0x03) << (2 * q));
        }
        out.push_back(static_cast<int8_t>(b));
      }
    }
  }
  return out;
}

// Deterministic weights covering the full signed range, both endpoints
// included.
std::vector<int8_t>
make_weights(const int64_t out_dim, const int64_t in_dim, const int64_t bits) {
  const int32_t lo = -(1 << (bits - 1));
  const int32_t span = 1 << bits;
  std::vector<int8_t> w(out_dim * in_dim);
  uint32_t state = 12345;
  for (auto& v : w) {
    state = state * 1103515245u + 12345u;
    v = static_cast<int8_t>(lo + static_cast<int32_t>((state >> 16) % span));
  }
  w[0] = static_cast<int8_t>(lo);
  w[1] = static_cast<int8_t>(lo + span - 1);
  return w;
}

struct Case {
  int64_t bits;
  int64_t in_dim;
  int64_t out_dim;
  bool scalar_qparams = false;
  int32_t weight_zero_point = 0;
};

// Activation row i is the zero point plus +/-1 at position i, so every output
// isolates a single weight: out[i][j] = scale_j * (bias_j +/- (w[j][i] - wzp))
// + out_zp. Running in_dim rows therefore checks every decoded weight exactly,
// and exercises leading_dims > 1.
void run_and_check(KernelRuntimeContext& ctx, const Case& c) {
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Int> tf_int;

  const std::vector<int8_t> dense = make_weights(c.out_dim, c.in_dim, c.bits);
  const std::vector<int8_t> packed = pack(dense, c.in_dim, c.bits);
  const int64_t row_bytes = c.in_dim * c.bits / 8;
  ASSERT_EQ(packed.size(), static_cast<size_t>(c.out_dim * row_bytes));

  // Split the rows across two leading dims so the input is rank 3.
  const int64_t rows = c.in_dim;
  std::vector<int8_t> in_values(rows * c.in_dim, kInZeroPoint);
  std::vector<int32_t> sign(rows);
  for (int64_t i = 0; i < rows; ++i) {
    sign[i] = (i % 2 == 0) ? 1 : -1;
    in_values[i * c.in_dim + i] = static_cast<int8_t>(kInZeroPoint + sign[i]);
  }

  std::vector<int32_t> bias(c.out_dim);
  std::vector<int32_t> shift(c.scalar_qparams ? 1 : c.out_dim);
  for (int64_t j = 0; j < c.out_dim; ++j) {
    bias[j] = static_cast<int32_t>(j % 41) - 20;
  }
  for (size_t j = 0; j < shift.size(); ++j) {
    shift[j] = c.scalar_qparams ? 2 : 1 + static_cast<int32_t>(j % 2);
  }
  const std::vector<int32_t> mult(shift.size(), kHalfMultiplier);

  std::vector<int8_t> expected(rows * c.out_dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < c.out_dim; ++j) {
      const int32_t s = shift[c.scalar_qparams ? 0 : j];
      const int32_t acc =
          bias[j] + sign[i] * (dense[j * c.in_dim + i] - c.weight_zero_point);
      const int32_t v = acc * (1 << (s - 1)) + kOutZeroPoint;
      expected[i * c.out_dim + j] =
          static_cast<int8_t>(std::min(127, std::max(-128, v)));
    }
  }

  const int32_t lead = static_cast<int32_t>(rows / 2);
  const int32_t in_dim = static_cast<int32_t>(c.in_dim);
  const int32_t out_dim = static_cast<int32_t>(c.out_dim);
  Tensor in = tf_char.make({2, lead, in_dim}, in_values);
  Tensor weight =
      tf_char.make({out_dim, static_cast<int32_t>(row_bytes)}, packed);
  Tensor bias_t = tf_int.make({out_dim}, bias);
  Tensor mult_t = tf_int.make({static_cast<int32_t>(mult.size())}, mult);
  Tensor shift_t = tf_int.make({static_cast<int32_t>(shift.size())}, shift);
  Tensor wzp_t = tf_int.make({1}, {c.weight_zero_point});
  Tensor out = tf_char.zeros({2, lead, out_dim});

  quantized_fully_connected_packed_out(
      ctx,
      in,
      weight,
      bias_t,
      c.in_dim,
      c.bits,
      kInZeroPoint,
      wzp_t,
      mult_t,
      shift_t,
      kOutZeroPoint,
      std::nullopt,
      out);

  const int8_t* got = out.const_data_ptr<int8_t>();
  int64_t mismatches = 0;
  for (size_t n = 0; n < expected.size(); ++n) {
    if (got[n] != expected[n] && ++mismatches <= 8) {
      ADD_FAILURE() << "row " << n / c.out_dim << ", channel " << n % c.out_dim
                    << ": got " << static_cast<int32_t>(got[n]) << ", want "
                    << static_cast<int32_t>(expected[n]);
    }
  }
  EXPECT_EQ(mismatches, 0);
}

// Large enough for the biggest tile any case here asks for.
alignas(16) uint8_t scratch_[32 * 1024];

class HiFiQuantizedFullyConnectedPackedTest : public OperatorTest {};

// in_dim = 32 takes the vector unpack; out_dim = 136 spans two row tiles
// (128 + 8), so the tile loop and the per-channel qparam offsets are covered.
TEST_F(HiFiQuantizedFullyConnectedPackedTest, VectorUnpack) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    MemoryAllocator allocator(sizeof(scratch_), scratch_);
    KernelRuntimeContext ctx(nullptr, &allocator);
    run_and_check(ctx, Case{bits, 32, 136});
  }
}

// in_dim = 40 is not a multiple of 32, so the scalar unpack serves the nnlib
// path.
TEST_F(HiFiQuantizedFullyConnectedPackedTest, ScalarUnpack) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    MemoryAllocator allocator(sizeof(scratch_), scratch_);
    KernelRuntimeContext ctx(nullptr, &allocator);
    run_and_check(ctx, Case{bits, 40, 24});
  }
}

// Length-1 qparams must be broadcast to every output channel.
TEST_F(HiFiQuantizedFullyConnectedPackedTest, ScalarQParams) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    MemoryAllocator allocator(sizeof(scratch_), scratch_);
    KernelRuntimeContext ctx(nullptr, &allocator);
    run_and_check(ctx, Case{bits, 32, 136, /*scalar_qparams=*/true});
  }
}

// No temp allocator: every scratch allocation fails and the kernel must fall
// back to the scalar path rather than abort. The scalar path is slow on the
// ISS, so these shapes are kept small.
TEST_F(HiFiQuantizedFullyConnectedPackedTest, NoScratchFallsBackToScalar) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    run_and_check(context_, Case{bits, 32, 8});
    run_and_check(context_, Case{bits, 32, 8, /*scalar_qparams=*/true});
  }
}

// nnlib's matmul has no weight zero-point operand, so an asymmetric weight
// must take the scalar path.
TEST_F(
    HiFiQuantizedFullyConnectedPackedTest,
    AsymmetricWeightFallsBackToScalar) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    MemoryAllocator allocator(sizeof(scratch_), scratch_);
    KernelRuntimeContext ctx(nullptr, &allocator);
    run_and_check(ctx, Case{bits, 32, 8, false, /*weight_zero_point=*/3});
  }
}

// Scratch for exactly one weight tile (128 rows x 32 for in_dim = 32) and
// nothing more: the bias tile allocation fails, so the offset-binary bias
// cannot be folded and the unpack has to subtract it per value instead.
TEST_F(HiFiQuantizedFullyConnectedPackedTest, NoBiasScratchSubtractsOffset) {
  for (int64_t bits : {4, 6}) {
    SCOPED_TRACE(bits);
    MemoryAllocator allocator(128 * 32, scratch_);
    KernelRuntimeContext ctx(nullptr, &allocator);
    run_and_check(ctx, Case{bits, 32, 136});
  }
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
