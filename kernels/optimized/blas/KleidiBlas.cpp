/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/KleidiBlas.h>

namespace executorch {
namespace cpublas {

using executorch::aten::BFloat16;

#ifdef ET_BUILD_WITH_KLEIDIAI
namespace {

static_assert(sizeof(BFloat16) == sizeof(uint16_t));

enum class KleidiBFloat16Backend { None, Neon, Sme2 };

// GEMM is column-major, while KleidiAI writes row-major. Compute
// C^T = op(B)^T @ op(A)^T: the packers' lhs is B, rhs is A, and (kai_m, kai_n)
// is (n, m). Both ukernels' get_dst_size returns m * n * sizeof(float), and
// their partial-tile stores stay within those dimensions; only inputs pad
// tiles.
#if defined(ET_KLEIDIAI_HAS_NEON_BF16) || defined(ET_KLEIDIAI_HAS_SME2_BF16)
constexpr int64_t kMinColumnsForKleidiai = 4;
using PackedBuffer = std::vector<uint64_t>;

size_t round_up(const size_t value, const size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

void resize_packed_buffer(PackedBuffer& buffer, const size_t size_bytes) {
  const size_t size = round_up(size_bytes, sizeof(uint64_t)) / sizeof(uint64_t);
  // Retain the high-water size so alternating SDPA tile shapes do not zero-fill
  // storage that the packers will overwrite.
  if (buffer.size() < size) {
    buffer.resize(size);
  }
}
#endif

KleidiBFloat16Backend select_kleidiai_bfloat16_backend(
    const TransposeType transb,
    const int64_t n) {
  (void)transb;
  (void)n;
  static const bool cpuinfo_initialized = cpuinfo_initialize();
  if (!cpuinfo_initialized) {
    return KleidiBFloat16Backend::None;
  }
#ifdef ET_KLEIDIAI_HAS_SME2_BF16
  if (transb == TransposeType::NoTranspose && n >= kMinColumnsForKleidiai &&
      cpuinfo_has_arm_sme2()) {
    return KleidiBFloat16Backend::Sme2;
  }
#endif
#ifdef ET_KLEIDIAI_HAS_NEON_BF16
  if (n >= kMinColumnsForKleidiai && cpuinfo_has_arm_bf16()) {
    return KleidiBFloat16Backend::Neon;
  }
#endif
  return KleidiBFloat16Backend::None;
}

#ifdef ET_KLEIDIAI_HAS_NEON_BF16
void write_bfloat16(uint8_t*& destination, const BFloat16 value) {
  std::memcpy(destination, &value, sizeof(value));
  destination += sizeof(value);
}

void pack_lhs_neon(
    const BFloat16* b,
    const int64_t ldb,
    const TransposeType transb,
    const size_t rows,
    const size_t k,
    const size_t mr,
    const size_t kr,
    void* packed_lhs) {
  auto* output = static_cast<uint8_t*>(packed_lhs);
  for (size_t row_block = 0; row_block < rows; row_block += mr) {
    for (size_t k_block = 0; k_block < k; k_block += kr) {
      for (size_t row_offset = 0; row_offset < mr; ++row_offset) {
        const size_t row = row_block + row_offset;
        for (size_t k_offset = 0; k_offset < kr; ++k_offset) {
          const size_t k_index = k_block + k_offset;
          BFloat16 value = 0;
          if (row < rows && k_index < k) {
            value = transb == TransposeType::NoTranspose
                ? b[row * ldb + k_index]
                : b[k_index * ldb + row];
          }
          write_bfloat16(output, value);
        }
      }
    }
  }
}

void pack_rhs_neon(
    const BFloat16* a,
    const int64_t lda,
    const TransposeType transa,
    const size_t rows,
    const size_t k,
    const size_t nr,
    const size_t kr,
    void* packed_rhs) {
  auto* output = static_cast<uint8_t*>(packed_rhs);
  for (size_t row_block = 0; row_block < rows; row_block += nr) {
    std::memset(output, 0, nr * sizeof(float));
    output += nr * sizeof(float);
    for (size_t k_block = 0; k_block < k; k_block += kr) {
      for (size_t row_offset = 0; row_offset < nr; ++row_offset) {
        const size_t row = row_block + row_offset;
        for (size_t k_offset = 0; k_offset < kr; ++k_offset) {
          const size_t k_index = k_block + k_offset;
          BFloat16 value = 0;
          if (row < rows && k_index < k) {
            value = transa == TransposeType::NoTranspose
                ? a[k_index * lda + row]
                : a[row * lda + k_index];
          }
          write_bfloat16(output, value);
        }
      }
    }
  }
}

void run_kleidiai_neon_bfloat16_gemm(
    const TransposeType transa,
    const TransposeType transb,
    const size_t m,
    const size_t n,
    const size_t k,
    const BFloat16* a,
    const int64_t lda,
    const BFloat16* b,
    const int64_t ldb,
    float* destination,
    const size_t destination_stride) {
  const size_t kai_m = n;
  const size_t kai_n = m;
  const size_t kai_k = k;
  const size_t mr =
      kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
  const size_t nr =
      kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
  const size_t kr =
      kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
  const size_t m_step =
      kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();

  /* library-local */ thread_local PackedBuffer lhs_packed;
  /* library-local */ thread_local PackedBuffer rhs_packed;
  resize_packed_buffer(
      lhs_packed,
      kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
          round_up(kai_m, mr), kai_k));
  resize_packed_buffer(
      rhs_packed,
      kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
          round_up(kai_n, nr), kai_k));
  pack_lhs_neon(b, ldb, transb, kai_m, kai_k, mr, kr, lhs_packed.data());
  pack_rhs_neon(a, lda, transa, kai_n, kai_k, nr, kr, rhs_packed.data());

  for (size_t m_index = 0; m_index < kai_m; m_index += m_step) {
    const size_t height = std::min(m_step, kai_m - m_index);
    kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
        height,
        kai_n,
        kai_k,
        reinterpret_cast<const uint8_t*>(lhs_packed.data()) +
            kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                m_index, kai_k),
        rhs_packed.data(),
        reinterpret_cast<uint8_t*>(destination) +
            kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                m_index, 0, destination_stride * sizeof(float)),
        destination_stride * sizeof(float),
        sizeof(float),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
  }
}
#endif

#ifdef ET_KLEIDIAI_HAS_SME2_BF16
void pack_rhs_sme2(
    const BFloat16* a,
    const int64_t lda,
    const TransposeType transa,
    const size_t rows,
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    void* packed_rhs) {
  for (size_t row = 0; row < rows; row += nr) {
    auto* packed_block = reinterpret_cast<uint8_t*>(packed_rhs) +
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa(
                             row, k);
    std::memset(packed_block, 0, nr * sizeof(float));
    // The x16 packer writes a 16-bit bias. Offset it into the zero-filled FP32
    // bias slot so the BF16 payload starts where the FP32 kernel expects it.
    void* x16_packed_block = packed_block + nr * sizeof(uint16_t);
    const size_t block_rows = std::min(nr, rows - row);
    if (transa == TransposeType::NoTranspose) {
      kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
          1,
          block_rows,
          k,
          nr,
          kr,
          sr,
          static_cast<size_t>(lda) * sizeof(BFloat16),
          a + row,
          packed_block,
          nullptr,
          x16_packed_block,
          0,
          nullptr);
    } else {
      kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
          1,
          block_rows,
          k,
          nr,
          kr,
          sr,
          static_cast<size_t>(lda) * sizeof(BFloat16),
          a + row * static_cast<size_t>(lda),
          packed_block,
          nullptr,
          x16_packed_block,
          0,
          nullptr);
    }
  }
}

void run_kleidiai_sme2_bfloat16_gemm(
    const TransposeType transa,
    const TransposeType transb,
    const size_t m,
    const size_t n,
    const size_t k,
    const BFloat16* a,
    const int64_t lda,
    const BFloat16* b,
    const int64_t ldb,
    float* destination,
    const size_t destination_stride) {
  // The SME LHS packer needs contiguous reduction elements in each row of B^T.
  ET_CHECK(transb == TransposeType::NoTranspose);
  const size_t kai_m = n;
  const size_t kai_n = m;
  const size_t kai_k = k;
  const size_t mr =
      kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();
  const size_t nr =
      kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();
  const size_t kr =
      kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();
  const size_t sr =
      kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();

  /* library-local */ thread_local PackedBuffer lhs_packed;
  /* library-local */ thread_local PackedBuffer rhs_packed;
  resize_packed_buffer(
      lhs_packed,
      kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(
          kai_m, kai_k, mr, kr, sr));
  resize_packed_buffer(
      rhs_packed,
      kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa(
          round_up(kai_n, nr), kai_k));

  kai_run_lhs_pack_x16p2vlx2_x16_sme(
      kai_m,
      kai_k,
      mr,
      kr,
      sr,
      0,
      b,
      static_cast<size_t>(ldb) * sizeof(BFloat16),
      lhs_packed.data());
  pack_rhs_sme2(a, lda, transa, kai_n, kai_k, nr, kr, sr, rhs_packed.data());

  kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa(
      kai_m,
      kai_n,
      kai_k,
      lhs_packed.data(),
      rhs_packed.data(),
      destination,
      destination_stride * sizeof(float),
      sizeof(float),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity());
}
#endif

} // namespace
#endif // ET_BUILD_WITH_KLEIDIAI

bool gemm_uses_kleidiai_bfloat16(const TransposeType transb, const int64_t n) {
#ifdef ET_BUILD_WITH_KLEIDIAI
  return select_kleidiai_bfloat16_backend(transb, n) !=
      KleidiBFloat16Backend::None;
#else
  (void)transb;
  (void)n;
  return false;
#endif
}

#if defined(ET_KLEIDIAI_HAS_NEON_BF16) || defined(ET_KLEIDIAI_HAS_SME2_BF16)
bool kleidiai_bfloat16_gemm(
    const TransposeType transa,
    const TransposeType transb,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const float alpha,
    const BFloat16* a,
    const int64_t lda,
    const BFloat16* b,
    const int64_t ldb,
    const float beta,
    float* c,
    const int64_t ldc) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return false;
  }
  const auto backend = select_kleidiai_bfloat16_backend(transb, n);
  if (backend == KleidiBFloat16Backend::None) {
    return false;
  }

  const size_t output_rows = m;
  const size_t output_columns = n;
  const size_t reduction_size = k;
  const size_t output_stride = ldc;
  const bool write_directly = alpha == 1.0f && beta == 0.0f;
  // Cache up to 1 MiB for repeated SDPA tiles. Larger GEMMs use call-local
  // storage so a single large request does not retain m * n floats per thread.
  constexpr size_t kMaxCachedProductElements = (1024 * 1024) / sizeof(float);
  /* library-local */ thread_local std::unique_ptr<float[]> cached_product;
  /* library-local */ thread_local size_t cached_product_size = 0;
  std::unique_ptr<float[]> large_product;
  float* destination = c;
  size_t destination_stride = ldc;
  if (!write_directly) {
    const size_t product_size = output_rows * output_columns;
    if (product_size <= kMaxCachedProductElements) {
      if (cached_product_size < product_size) {
        cached_product.reset(new float[product_size]);
        cached_product_size = product_size;
      }
      destination = cached_product.get();
    } else {
      large_product.reset(new float[product_size]);
      destination = large_product.get();
    }
    destination_stride = output_rows;
  }

#ifdef ET_KLEIDIAI_HAS_SME2_BF16
  if (backend == KleidiBFloat16Backend::Sme2) {
    run_kleidiai_sme2_bfloat16_gemm(
        transa,
        transb,
        output_rows,
        output_columns,
        reduction_size,
        a,
        lda,
        b,
        ldb,
        destination,
        destination_stride);
  }
#endif
#ifdef ET_KLEIDIAI_HAS_NEON_BF16
  if (backend == KleidiBFloat16Backend::Neon) {
    run_kleidiai_neon_bfloat16_gemm(
        transa,
        transb,
        output_rows,
        output_columns,
        reduction_size,
        a,
        lda,
        b,
        ldb,
        destination,
        destination_stride);
  }
#endif

  if (!write_directly) {
    for (size_t column = 0; column < output_columns; ++column) {
      for (size_t row = 0; row < output_rows; ++row) {
        const size_t output_index = column * output_stride + row;
        const float value = alpha * destination[column * output_rows + row];
        c[output_index] = beta == 0.0f ? value : value + beta * c[output_index];
      }
    }
  }
  return true;
}
#endif

} // namespace cpublas
} // namespace executorch
