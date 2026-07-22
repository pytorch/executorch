/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_moe.h>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <torchao/csrc/cpu/shared_kernels/internal/packed_weights_header.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>

#ifdef ENABLE_QUANTIZED_MOE_FFN
#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/kernel_selector.h>
#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <optional> // std::nullopt, used only by the optimized aarch64 path
#endif // ENABLE_QUANTIZED_MOE_FFN

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace torch {
namespace executor {
namespace native {

namespace {

using executorch::aten::string_view;

// Stable softmax over a row of length E (in place).
inline void softmax_row(float* row, int64_t e) {
  float maxv = row[0];
  for (int64_t i = 1; i < e; ++i) {
    if (row[i] > maxv) {
      maxv = row[i];
    }
  }
  float sum = 0.0f;
  for (int64_t i = 0; i < e; ++i) {
    row[i] = std::exp(row[i] - maxv);
    sum += row[i];
  }
  const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
  for (int64_t i = 0; i < e; ++i) {
    row[i] *= inv_sum;
  }
}

// In-place sigmoid over a contiguous buffer.
inline void sigmoid_inplace(float* p, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    p[i] = 1.0f / (1.0f + std::exp(-p[i]));
  }
}

// Simple O(E*K) top-k selection.  E and K are both small in MoE models
// (E <= 128, K <= 8), so this is cheaper than partial_sort and needs no
// scratch buffer.  Stable across ties via index ordering.
inline void
topk_indices(const float* scores, int64_t e, int64_t k, int32_t* out_indices) {
  for (int64_t ki = 0; ki < k; ++ki) {
    int32_t best = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int64_t ei = 0; ei < e; ++ei) {
      bool already_selected = false;
      for (int64_t j = 0; j < ki; ++j) {
        if (out_indices[j] == static_cast<int32_t>(ei)) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }
      if (scores[ei] > best_score ||
          (scores[ei] == best_score && (best < 0 || ei < best))) {
        best_score = scores[ei];
        best = static_cast<int32_t>(ei);
      }
    }
    // Non-finite (NaN) scores compare false against everything, so `best` can
    // remain -1 when fewer than `k` finite candidates are left. Writing -1
    // would later index cursor[-1]/expert_offsets out of bounds, so fall back
    // to the lowest-indexed unselected expert (num_activated <= num_experts
    // guarantees one exists) to keep the selected id valid.
    if (best < 0) {
      for (int64_t ei = 0; ei < e; ++ei) {
        bool already_selected = false;
        for (int64_t j = 0; j < ki; ++j) {
          if (out_indices[j] == static_cast<int32_t>(ei)) {
            already_selected = true;
            break;
          }
        }
        if (!already_selected) {
          best = static_cast<int32_t>(ei);
          break;
        }
      }
    }
    out_indices[ki] = best;
  }
}

// Reference linear: unpack torchao blob → dequantize → cpublas::gemm.
// TODO: move to torchao?
template <int kWeightNbit>
inline void reference_linear(
    const uint8_t* packed_w_blob,
    int64_t packed_blob_bytes,
    const float* x,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t group_size,
    float* out) {
  constexpr int kNr = 8, kKr = 16, kSr = 2;
  const auto header_size =
      static_cast<int64_t>(torchao::ops::PackedWeightsHeader::size());
  ET_CHECK_MSG(
      packed_blob_bytes >= header_size,
      "torchao packed blob too small to contain header");

  const void* packed_data = packed_w_blob + header_size;
  const auto n_int = static_cast<int>(n);
  const auto k_int = static_cast<int>(k);
  const auto gs_int = static_cast<int>(group_size);

  std::vector<int8_t> qvals(static_cast<size_t>(n * k));
  std::vector<float> scales(static_cast<size_t>(n * (k / group_size)));
  torchao::weight_packing::unpack_weights<kWeightNbit, kNr, kKr, kSr>(
      qvals.data(),
      scales.data(),
      /*weight_zeros=*/nullptr,
      /*bias=*/nullptr,
      n_int,
      k_int,
      gs_int,
      /*has_weight_zeros=*/false,
      /*has_bias=*/false,
      packed_data);

  std::vector<float> w_fp32(static_cast<size_t>(n * k));
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ki = 0; ki < k; ++ki) {
      const int64_t group_idx = ni * (k / group_size) + ki / group_size;
      w_fp32[static_cast<size_t>(ni * k + ki)] =
          static_cast<float>(qvals[static_cast<size_t>(ni * k + ki)]) *
          scales[static_cast<size_t>(group_idx)];
    }
  }

  // w_fp32 is [N, K] row-major.  We need out = x [M, K] @ w^T [K, N].
  ::executorch::cpublas::gemm(
      ::executorch::cpublas::TransposeType::Transpose,
      ::executorch::cpublas::TransposeType::NoTranspose,
      /*m=*/n,
      /*n=*/m,
      /*k=*/k,
      /*alpha=*/1.0f,
      /*a=*/w_fp32.data(),
      /*lda=*/k,
      /*b=*/x,
      /*ldb=*/k,
      /*beta=*/0.0f,
      /*c=*/out,
      /*ldc=*/n);
}

// Dispatch a single per-expert grouped GEMM through torchao's
// linear_operator (optimized, aarch64) or reference unpack+dequant+gemm.
#ifdef ENABLE_QUANTIZED_MOE_FFN
template <int kWeightNbit>
inline void torchao_linear(
    const uint8_t* packed_w_blob,
    int64_t packed_blob_bytes,
    const float* x,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t group_size,
    float* out) {
  ET_CHECK_MSG(
      packed_blob_bytes >=
          static_cast<int64_t>(torchao::ops::PackedWeightsHeader::size()),
      "torchao packed blob too small to contain header");
  auto header = torchao::ops::PackedWeightsHeader::read(packed_w_blob);
  auto uk = torchao::ops::linear_8bit_act_xbit_weight::select_ukernel_config<
      kWeightNbit>(header);

  torchao::ops::linear_8bit_act_xbit_weight::linear_operator(
      uk,
      /*tiling_params=*/std::nullopt,
      /*output=*/out,
      /*m=*/static_cast<int>(m),
      /*n=*/static_cast<int>(n),
      /*k=*/static_cast<int>(k),
      /*group_size=*/static_cast<int>(group_size),
      /*packed_weights=*/packed_w_blob +
          torchao::ops::PackedWeightsHeader::size(),
      /*activations=*/x,
      /*has_clamp=*/false,
      /*clamp_min=*/0.0f,
      /*clamp_max=*/0.0f);
}
#endif // ENABLE_QUANTIZED_MOE_FFN

inline void expert_linear_dispatch(
    int64_t weight_nbit,
    const uint8_t* packed_w_blob,
    int64_t packed_blob_bytes,
    const float* x,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t group_size,
    float* out) {
  // Validate the blob holds at least the header plus the torchao packed
  // weight-data bytes for the claimed dims before any path dereferences it.
  constexpr int kNr = 8, kKr = 16, kSr = 2;
  const int64_t required_bytes =
      static_cast<int64_t>(torchao::ops::PackedWeightsHeader::size()) +
      static_cast<int64_t>(torchao::weight_packing::packed_weights_size(
          static_cast<int>(n),
          static_cast<int>(k),
          static_cast<int>(group_size),
          static_cast<int>(weight_nbit),
          /*has_weight_zeros=*/false,
          /*has_bias=*/false,
          kNr,
          kKr,
          kSr));
  ET_CHECK_MSG(
      packed_blob_bytes >= required_bytes,
      "torchao packed blob too small: have %lld bytes, need >= %lld for "
      "(n=%lld, k=%lld, group_size=%lld, weight_nbit=%lld)",
      static_cast<long long>(packed_blob_bytes),
      static_cast<long long>(required_bytes),
      static_cast<long long>(n),
      static_cast<long long>(k),
      static_cast<long long>(group_size),
      static_cast<long long>(weight_nbit));
  switch (weight_nbit) {
    case 4:
#ifdef ENABLE_QUANTIZED_MOE_FFN
      torchao_linear<4>(
          packed_w_blob, packed_blob_bytes, x, m, n, k, group_size, out);
#else
      reference_linear<4>(
          packed_w_blob, packed_blob_bytes, x, m, n, k, group_size, out);
#endif
      return;
    case 8:
#ifdef ENABLE_QUANTIZED_MOE_FFN
      torchao_linear<8>(
          packed_w_blob, packed_blob_bytes, x, m, n, k, group_size, out);
#else
      reference_linear<8>(
          packed_w_blob, packed_blob_bytes, x, m, n, k, group_size, out);
#endif
      return;
    default:
      ET_CHECK_MSG(
          false,
          "quantized_moe_ffn: unsupported weight_nbit=%lld",
          static_cast<long long>(weight_nbit));
  }
}

} // namespace

Tensor& quantized_moe_ffn_out(
    KernelRuntimeContext& ctx,
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& expert_bias,
    const Tensor& packed_w13,
    const Tensor& packed_w2,
    int64_t num_activated_experts,
    int64_t num_experts,
    int64_t hidden_dim,
    int64_t dim,
    int64_t group_size,
    int64_t weight_nbit,
    string_view score_func,
    double route_scale,
    Tensor& out) {
  (void)ctx;

  // ----- Shape & dtype checks -----
  ET_CHECK_MSG(x.dim() == 2, "x must be 2D [T, D]");
  ET_CHECK_MSG(
      x.size(1) == dim,
      "x last dim must equal dim (got %lld vs %lld)",
      static_cast<long long>(x.size(1)),
      static_cast<long long>(dim));
  ET_CHECK_MSG(x.scalar_type() == ScalarType::Float, "x must be fp32");

  ET_CHECK_MSG(
      gate_weight.dim() == 2 && gate_weight.size(0) == num_experts &&
          gate_weight.size(1) == dim,
      "gate_weight must be [E, D]");
  ET_CHECK_MSG(
      gate_weight.scalar_type() == ScalarType::Float,
      "gate_weight must be fp32");

  const bool use_expert_bias = expert_bias.numel() > 0;
  if (use_expert_bias) {
    ET_CHECK_MSG(
        expert_bias.dim() == 1 && expert_bias.size(0) == num_experts,
        "expert_bias must be [E] when provided");
    ET_CHECK_MSG(
        expert_bias.scalar_type() == ScalarType::Float,
        "expert_bias must be fp32");
  }

  ET_CHECK_MSG(
      packed_w13.dim() == 2 && packed_w13.size(0) == num_experts,
      "packed_w13 must be [E, packed_bytes]");
  ET_CHECK_MSG(
      packed_w2.dim() == 2 && packed_w2.size(0) == num_experts,
      "packed_w2 must be [E, packed_bytes]");
  ET_CHECK_MSG(
      packed_w13.scalar_type() == ScalarType::Byte, "packed_w13 must be uint8");
  ET_CHECK_MSG(
      packed_w2.scalar_type() == ScalarType::Byte, "packed_w2 must be uint8");

  ET_CHECK_MSG(
      num_activated_experts > 0 && num_activated_experts <= num_experts,
      "num_activated_experts out of range");

  ET_CHECK_MSG(
      score_func == "sigmoid" || score_func == "softmax",
      "score_func must be \"sigmoid\" or \"softmax\"");
  // expert_bias only shifts sigmoid top-k selection; the softmax path never
  // reads it, so reject the combination instead of silently ignoring it.
  ET_CHECK_MSG(
      !(use_expert_bias && score_func == "softmax"),
      "expert_bias is only supported with score_func=\"sigmoid\"");

  ET_CHECK_MSG(group_size > 0, "group_size must be positive");
  ET_CHECK_MSG(dim > 0, "dim must be positive");
  ET_CHECK_MSG(hidden_dim > 0, "hidden_dim must be positive");
  ET_CHECK_MSG(
      dim % group_size == 0,
      "dim (%lld) must be divisible by group_size (%lld)",
      static_cast<long long>(dim),
      static_cast<long long>(group_size));
  ET_CHECK_MSG(
      hidden_dim % group_size == 0,
      "hidden_dim (%lld) must be divisible by group_size (%lld)",
      static_cast<long long>(hidden_dim),
      static_cast<long long>(group_size));

  const int64_t T = x.size(0);
  const int64_t D = dim;
  const int64_t F = hidden_dim;
  const int64_t E = num_experts;
  const int64_t K = num_activated_experts;
  const int64_t pw13_bytes = packed_w13.size(1);
  const int64_t pw2_bytes = packed_w2.size(1);

  // Resize the output to [T, D].
  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == ScalarType::Float,
      InvalidArgument,
      out,
      "output must be fp32");
  Tensor::SizesType expected_out_sizes[2] = {
      static_cast<Tensor::SizesType>(T), static_cast<Tensor::SizesType>(D)};
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_sizes, 2}) == Error::Ok,
      InvalidArgument,
      out);

  const float* x_ptr = x.const_data_ptr<float>();
  const float* gate_w_ptr = gate_weight.const_data_ptr<float>();
  const float* expert_bias_ptr =
      use_expert_bias ? expert_bias.const_data_ptr<float>() : nullptr;
  const uint8_t* pw13_ptr = packed_w13.const_data_ptr<uint8_t>();
  const uint8_t* pw2_ptr = packed_w2.const_data_ptr<uint8_t>();
  float* out_ptr = out.mutable_data_ptr<float>();

  // Zero the output: we'll do a weighted scatter-add into it at the end.
  std::memset(out_ptr, 0, sizeof(float) * static_cast<size_t>(T * D));

  // ----- 1. Router GEMM: scores [T, E] = x [T, D] @ gate_weight^T -----
  //
  // Row-major equivalent of `cpublas::gemm` arguments: see the
  // op_sdpa_impl.h comment about column-major BLAS interpretation. With
  // `transa=Transpose, transb=NoTranspose, m=E, n=T, k=D`, the call
  // produces `scores [T, E]` row-major.
  std::vector<float> scores(static_cast<size_t>(T * E), 0.0f);
  ::executorch::cpublas::gemm(
      ::executorch::cpublas::TransposeType::Transpose, // transa: gate_weight
      ::executorch::cpublas::TransposeType::NoTranspose, // transb: x
      /*m=*/E,
      /*n=*/T,
      /*k=*/D,
      /*alpha=*/1.0f,
      /*a=*/gate_w_ptr,
      /*lda=*/D,
      /*b=*/x_ptr,
      /*ldb=*/D,
      /*beta=*/0.0f,
      /*c=*/scores.data(),
      /*ldc=*/E);

  // ----- 2/3. Score gating + top-k -----
  //
  // We allocate per-token expert indices [T, K] and per-token unbiased
  // routing weights [T, K]. For sigmoid: scores are sigmoid'd in place,
  // top-k selection uses (scores + bias) but the gathered weights come
  // from the un-biased scores, then are renormalized by row-sum and
  // multiplied by route_scale. For softmax: top-k of raw scores, then
  // softmax over the k gathered values.
  std::vector<int32_t> expert_indices(static_cast<size_t>(T * K), 0);
  std::vector<float> expert_weights(static_cast<size_t>(T * K), 0.0f);

  const bool is_sigmoid = (score_func == "sigmoid");
  if (is_sigmoid) {
    sigmoid_inplace(scores.data(), T * E);
  }
  std::vector<float> scores_for_topk;
  if (is_sigmoid && use_expert_bias) {
    scores_for_topk.assign(scores.begin(), scores.end());
    for (int64_t t = 0; t < T; ++t) {
      float* row = scores_for_topk.data() + t * E;
      for (int64_t e = 0; e < E; ++e) {
        row[e] += expert_bias_ptr[e];
      }
    }
  }
  const float* topk_src =
      scores_for_topk.empty() ? scores.data() : scores_for_topk.data();

  for (int64_t t = 0; t < T; ++t) {
    int32_t* idx_row = expert_indices.data() + t * K;
    float* w_row = expert_weights.data() + t * K;

    topk_indices(topk_src + t * E, E, K, idx_row);

    if (is_sigmoid) {
      // Gather UN-biased sigmoid scores, then renormalize by row-sum and
      // scale by route_scale.
      float row_sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float v = scores[t * E + idx_row[k]];
        w_row[k] = v;
        row_sum += v;
      }
      const float scale = static_cast<float>(route_scale) / (row_sum + 1e-20f);
      for (int64_t k = 0; k < K; ++k) {
        w_row[k] *= scale;
      }
    } else {
      // Softmax path: gather pre-softmax scores at the top-k indices,
      // then softmax the K-vector.
      for (int64_t k = 0; k < K; ++k) {
        w_row[k] = scores[t * E + idx_row[k]];
      }
      softmax_row(w_row, K);
    }
  }

  // ----- 4. Permute (counting sort over flattened [T*K] expert IDs) -----
  std::vector<int64_t> expert_offsets(static_cast<size_t>(E + 1), 0);
  const int64_t total_pairs = T * K;
  for (int64_t i = 0; i < total_pairs; ++i) {
    ++expert_offsets[expert_indices[i] + 1];
  }
  for (int64_t e = 0; e < E; ++e) {
    expert_offsets[e + 1] += expert_offsets[e];
  }
  // permuted_token_idx[r] = original token index that produced gathered row r.
  // permuted_gate[r]      = routing weight to apply on the way out.
  std::vector<int32_t> permuted_token_idx(static_cast<size_t>(total_pairs), 0);
  std::vector<float> permuted_gate(static_cast<size_t>(total_pairs), 0.0f);
  std::vector<int64_t> cursor(static_cast<size_t>(E), 0);
  for (int64_t e = 0; e < E; ++e) {
    cursor[e] = expert_offsets[e];
  }
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t k = 0; k < K; ++k) {
      const int32_t e = expert_indices[t * K + k];
      const int64_t pos = cursor[e]++;
      permuted_token_idx[pos] = static_cast<int32_t>(t);
      permuted_gate[pos] = expert_weights[t * K + k];
    }
  }

  // ----- 5. Gather x_perm [total_pairs, D] -----
  // Temporary buffers scale with T*K*D + max_m_e*(2F+D) floats.
  // For typical MoE decode (T=1, K=4, D=768, F=384): ~15 KB total.
  std::vector<float> x_perm(static_cast<size_t>(total_pairs * D), 0.0f);
  for (int64_t r = 0; r < total_pairs; ++r) {
    const int32_t t = permuted_token_idx[r];
    std::memcpy(
        x_perm.data() + r * D,
        x_ptr + static_cast<int64_t>(t) * D,
        sizeof(float) * static_cast<size_t>(D));
  }

  // ----- 6. Per-expert grouped GEMM via torchao -----
  // Reusable per-expert scratch (sized for the worst case max_m_e).
  int64_t max_m_e = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t m_e = expert_offsets[e + 1] - expert_offsets[e];
    if (m_e > max_m_e) {
      max_m_e = m_e;
    }
  }
  // h13_buf holds the fused [m_e, 2F] output of the w1+w3 GEMM.
  // mid_buf holds the [m_e, F] SwiGLU result compacted for the w2 GEMM.
  std::vector<float> h13_buf(static_cast<size_t>(max_m_e * 2 * F), 0.0f);
  std::vector<float> mid_buf(static_cast<size_t>(max_m_e * F), 0.0f);
  std::vector<float> out_e_buf(static_cast<size_t>(max_m_e * D), 0.0f);

  for (int64_t e = 0; e < E; ++e) {
    const int64_t m_e = expert_offsets[e + 1] - expert_offsets[e];
    if (m_e == 0) {
      continue;
    }
    const float* x_e = x_perm.data() + expert_offsets[e] * D;
    const uint8_t* w13_e = pw13_ptr + e * pw13_bytes;
    const uint8_t* w2_e = pw2_ptr + e * pw2_bytes;

    // Fused up+gate projection (D -> 2F): H13 = x_e @ w13[e]^T
    // First F columns are h1 (up), last F are h3 (gate).
    expert_linear_dispatch(
        weight_nbit,
        w13_e,
        pw13_bytes,
        x_e,
        m_e,
        2 * F,
        D,
        group_size,
        h13_buf.data());
    // SwiGLU + compact: read [m_e, 2F] interleaved, write [m_e, F]
    for (int64_t i = 0; i < m_e; ++i) {
      const float* src = h13_buf.data() + i * 2 * F;
      float* dst = mid_buf.data() + i * F;
      for (int64_t j = 0; j < F; ++j) {
        const float v = src[j];
        const float s = v / (1.0f + std::exp(-v));
        dst[j] = s * src[F + j];
      }
    }
    // Down-projection (F -> D): OUT_e = MID @ w2[e]^T
    expert_linear_dispatch(
        weight_nbit,
        w2_e,
        pw2_bytes,
        mid_buf.data(),
        m_e,
        D,
        F,
        group_size,
        out_e_buf.data());

    // ----- 7. Weighted scatter-add unpermute for this expert's block -----
    for (int64_t r_local = 0; r_local < m_e; ++r_local) {
      const int64_t r = expert_offsets[e] + r_local;
      const int32_t t = permuted_token_idx[r];
      const float gate = permuted_gate[r];
      float* dst = out_ptr + static_cast<int64_t>(t) * D;
      const float* src = out_e_buf.data() + r_local * D;
      for (int64_t d = 0; d < D; ++d) {
        dst[d] += gate * src[d];
      }
    }
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "quantized_moe_ffn.out",
    torch::executor::native::quantized_moe_ffn_out);
