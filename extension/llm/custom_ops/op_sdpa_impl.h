/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec_n.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <vector>

#ifdef ET_USE_THREADPOOL
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#endif
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

#include <torchao/csrc/cpu/torch_free_kernels/interface/quantized_matmul.h>

namespace torch {
namespace executor {

namespace native {

enum class SeqDim { ONE = 1, TWO };

namespace sdpa::impl {

struct MaybeQuantizedMatrixData {
  const void* data{nullptr};
  const int8_t* zero_points{nullptr};
  const float* scales{nullptr};
  int64_t m = 0, n = 0;
  const int64_t zero_points_stride{1};
  const int64_t scales_stride{1};
  ScalarType dtype{ScalarType::Float};
  MaybeQuantizedMatrixData() = default;
  MaybeQuantizedMatrixData(
      const void* data_,
      const int8_t* zero_points_,
      const float* scales_,
      int64_t m_,
      int64_t n_,
      int64_t qparams_stride,
      ScalarType dtype_)
      : data(data_),
        zero_points(zero_points_),
        scales(scales_),
        m(m_),
        n(n_),
        zero_points_stride(qparams_stride),
        scales_stride(qparams_stride),
        dtype(dtype_) {}
};

template <typename accum_t>
void _q_at_k_gemm(
    const int64_t q_m,
    const int64_t k_n,
    const int64_t qk_k,
    const MaybeQuantizedMatrixData& q_data,
    const int64_t q_stride_m,
    const MaybeQuantizedMatrixData& k_data,
    const int64_t k_stride_n,
    accum_t* qk_data) {
  ET_CHECK_MSG(q_data.dtype == k_data.dtype, "q and k must have same dtype");
  ET_CHECK_MSG(
      q_data.dtype == ScalarType::Char || q_data.dtype == ScalarType::Float,
      "q and k must be either int8 or float");
  if (q_data.dtype == ScalarType::Char) {
    if constexpr (std::is_same<accum_t, float>::value) {
      int a_stride_m_tmp, b_stride_n_tmp;
      auto kernel = torchao::kernels::cpu::quantized_matmul::
          get_int8_a_int8_b_channelwise_qmatmul(
              q_m, k_n, qk_k, false, true, a_stride_m_tmp, b_stride_n_tmp);
      kernel(
          q_m,
          k_n,
          qk_k,
          static_cast<const int8_t*>(q_data.data),
          q_stride_m,
          static_cast<const int8_t*>(k_data.data),
          k_stride_n,
          qk_data,
          k_n,
          static_cast<const int8_t*>(q_data.zero_points),
          static_cast<const int8_t*>(k_data.zero_points),
          static_cast<const float*>(q_data.scales),
          static_cast<const float*>(k_data.scales),
          // LHS and RHS are assumed to have same stride for qparams
          q_data.zero_points_stride,
          k_data.zero_points_stride);
    } else {
      ET_CHECK_MSG(
          false, "Accumulation in dtype other than float not supported yet");
    }
  } else {
    ::executorch::cpublas::gemm(
        ::executorch::cpublas::TransposeType::Transpose,
        ::executorch::cpublas::TransposeType::NoTranspose,
        k_n,
        q_m,
        qk_k,
        static_cast<accum_t>(1),
        static_cast<const accum_t*>(k_data.data),
        k_stride_n,
        static_cast<const accum_t*>(q_data.data),
        q_stride_m,
        static_cast<accum_t>(0),
        qk_data,
        k_n);
  }
}

// Refactor op_dequantize.cpp to avoid code duplication
void dequantize_optimized(
    const int8_t* in,
    const float scale,
    const int8_t zero_point,
    float* out,
    int64_t quant_min,
    int64_t quant_max,
    size_t numel) {
  size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
  int8x8_t zero_point_vec = vdup_n_s8(zero_point);
  float32x4_t scales = vdupq_n_f32(static_cast<float>(scale));
  constexpr int32_t kVecSize = 16;
  const size_t num_vecs = numel / kVecSize;
  const int8_t* in_copy = in;
  float* out_copy = out;
  for (; i < num_vecs; i++) {
    int8x16_t in_vec = vld1q_s8(in_copy);
    int16x8_t sub_vec_0_7 = vsubl_s8(vget_low_s8(in_vec), zero_point_vec);
    int32x4_t sub_vec_0_3 = vmovl_s16(vget_low_s16(sub_vec_0_7));
    int32x4_t sub_vec_4_7 = vmovl_s16(vget_high_s16(sub_vec_0_7));
    float32x4_t out_vec_0_3 = vmulq_f32(vcvtq_f32_s32(sub_vec_0_3), scales);
    float32x4_t out_vec_4_7 = vmulq_f32(vcvtq_f32_s32(sub_vec_4_7), scales);

    int16x8_t sub_vec_8_15 = vsubl_s8(vget_high_s8(in_vec), zero_point_vec);
    int32x4_t sub_vec_8_11 = vmovl_s16(vget_low_s16(sub_vec_8_15));
    int32x4_t sub_vec_12_15 = vmovl_s16(vget_high_s16(sub_vec_8_15));
    float32x4_t out_vec_8_11 = vmulq_f32(vcvtq_f32_s32(sub_vec_8_11), scales);
    float32x4_t out_vec_12_15 = vmulq_f32(vcvtq_f32_s32(sub_vec_12_15), scales);
    vst1q_f32(out_copy + 0, out_vec_0_3);
    vst1q_f32(out_copy + 4, out_vec_4_7);
    vst1q_f32(out_copy + 8, out_vec_8_11);
    vst1q_f32(out_copy + 12, out_vec_12_15);
    in_copy += kVecSize;
    out_copy += kVecSize;
  }
  i = i * kVecSize;
#endif
  for (; i < numel; i++) {
    out[i] = (static_cast<int16_t>(in[i]) - static_cast<int16_t>(zero_point)) *
        scale;
  }
}

void dequantize_per_channel_optimized(
    const int8_t* in_data,
    const float* scales_data,
    const int8_t* zero_points_data,
    float* out_data,
    int64_t quant_min,
    int64_t quant_max,
    size_t outer_size,
    size_t in_outer_stride,
    size_t out_outer_stride,
    size_t num_channels,
    size_t in_channel_stride,
    size_t out_channel_stride,
    size_t channel_size,
    size_t qparams_stride) {
  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    // Loop through dim
    for (size_t channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
      const int8_t* in_data_local = in_data + outer_idx * in_outer_stride +
          channel_idx * in_channel_stride;
      const float scale = *(scales_data + channel_idx * qparams_stride);
      const int8_t zero_point =
          *(zero_points_data + channel_idx * qparams_stride);
      float* out_data_local = out_data + outer_idx * out_outer_stride +
          channel_idx * out_channel_stride;
      dequantize_optimized(
          in_data_local,
          scale,
          zero_point,
          out_data_local,
          quant_min,
          quant_max,
          channel_size);
    }
  }
}

void dequant_and_gemm(
    const int64_t m,
    const int64_t n,
    const int64_t k,
    float* qk_data,
    const int64_t qk_stride_m,
    const MaybeQuantizedMatrixData& v_data,
    const int64_t v_stride_n,
    float* o_data,
    const int64_t o_stride_m,
    const float beta) {
  std::vector<float> dequantized_v_data(v_data.m * v_data.n);
  dequantize_per_channel_optimized(
      static_cast<const int8_t*>(v_data.data),
      static_cast<const float*>(v_data.scales),
      static_cast<const int8_t*>(v_data.zero_points),
      dequantized_v_data.data(),
      -128,
      127,
      1,
      0,
      0,
      v_data.m,
      v_stride_n,
      v_data.n,
      v_data.n,
      v_data.zero_points_stride);
  ::executorch::cpublas::gemm(
      ::executorch::cpublas::TransposeType::NoTranspose,
      ::executorch::cpublas::TransposeType::NoTranspose,
      n,
      m,
      k,
      static_cast<float>(1),
      dequantized_v_data.data(),
      v_data.n,
      qk_data,
      qk_stride_m,
      beta,
      o_data,
      o_stride_m);
}

template <typename accum_t>
void _qk_at_v_gemm(
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const accum_t* qk_data,
    const int64_t qk_stride_m,
    const MaybeQuantizedMatrixData& v_data,
    const int64_t v_stride_n,
    accum_t* o_data,
    const int64_t o_stride_m,
    const accum_t beta) {
  if (v_data.dtype == ScalarType::Char) {
    if constexpr (std::is_same<accum_t, float>::value) {
      if (m > 4) {
        // For larger batch sizes, dequantize and use BLAS for better
        // performance
        dequant_and_gemm(
            m,
            n,
            k,
            const_cast<float*>(qk_data),
            qk_stride_m,
            v_data,
            v_stride_n,
            o_data,
            o_stride_m,
            beta);
      } else {
        // For smaller batch sizes, use quantized gemm
        int a_stride_m_tmp, b_stride_n_tmp;
        auto kernel = torchao::kernels::cpu::quantized_matmul::
            get_fp32_a_input_channelwise_8bit_b_f32_c_matmul(
                m, n, k, false, false, a_stride_m_tmp, b_stride_n_tmp);
        kernel(
            m,
            n,
            k,
            qk_data,
            qk_stride_m /*lhs_stride_m*/,
            static_cast<const int8_t*>(v_data.data),
            v_stride_n /*rhs_stride_n*/,
            o_data,
            o_stride_m /*out_stride_n*/,
            static_cast<const int8_t*>(v_data.zero_points),
            static_cast<const float*>(v_data.scales),
            beta,
            v_data.zero_points_stride);
      }
    } else {
      ET_CHECK_MSG(
          false, "Accumulation in dtype other than float not supported yet");
    }
  } else {
    ::executorch::cpublas::gemm(
        ::executorch::cpublas::TransposeType::NoTranspose,
        ::executorch::cpublas::TransposeType::NoTranspose,
        n,
        m,
        k,
        static_cast<accum_t>(1),
        static_cast<const accum_t*>(v_data.data),
        v_stride_n,
        qk_data,
        qk_stride_m,
        beta,
        o_data,
        o_stride_m);
  }
}

constexpr size_t kKVDim = 4;

template <typename T>
inline void _store(T* dst, ::at::vec::Vectorized<T> src) {
  src.store(dst);
}

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

inline double calculate_scale(
    const Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale =
      scale.has_value() ? scale.value() : 1.0 / std::sqrt(query.size(3));
  return softmax_scale;
}

namespace vec = ::at::vec;
using Tensor = ::executorch::aten::Tensor;

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void
_exp_reduce_sum_fusion_kernel(T1* a, const int& size, T2* out, T1& val) {
  // NOTE: we observed numerics issues with this function when
  // deleting the old executorch::vec and replacing with at::vec
  // here. The major known difference is that executorch::vec was 256
  // bits wide vs 128 bits for at::vec (and the hardware). Preserving
  // this function's execution width at 256 bits and avoiding
  // vec_reduce_all below removed the issues.
  constexpr auto vec_size = vec::Vectorized<T1>::size() * 2;
  auto vec_max = vec::VectorizedN<T1, 2>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = vec::VectorizedN<T1, 2>(tmp_sum);
  for (int i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::VectorizedN<T1, 2>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    // Replace with exp_u20 later
    // auto tmp2 = tmp1.exp_u20();
    auto tmp2 = tmp1.exp();
    vec_tmp_sum = vec_tmp_sum + tmp2;
    tmp2.store(out + i);
  }

  __at_align__ T1 vec_tmp_sum_array[vec_size];
  vec_tmp_sum.store(vec_tmp_sum_array);
  for (const auto i : c10::irange(vec_size)) {
    tmp_sum += vec_tmp_sum_array[i];
  }
  // See NOTE above; we should replace the scalar reduction above with
  // this reduction (which uses vaddvq_f32 internally), but it changes
  // numerics.
  // tmp_sum = vec::vec_reduce_all<T1>(
  //     [](vec::Vectorized<T1>& x, vec::Vectorized<T1>& y) { return x + y; },
  //     vec_tmp_sum);
  for (int i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = vec::Vectorized<scalar_t>::size();
  auto vec_scale = vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = vec::Vectorized<scalar_t>(tmp_max);
  for (int i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (int i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      vec::vec_reduce_all<scalar_t>(
          [](vec::Vectorized<scalar_t>& x, vec::Vectorized<scalar_t>& y) {
            return vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  ET_CHECK(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<
        ::executorch::runtime::is_reduced_floating_point_v<scalar_t>,
        int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  (void)ptr;
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  for (; d < size; d++) {
    data[d] = val;
  }
}

/*
Note on start_pos as a parameter:
What is start_pos?
- start_pos is the position of the first element of the current query. That is,
in LLMs during generate phase, when we generate one token a time, the query
will correspond to monotonically increasing start_pos. e.g. the first token
is at start_pos = 0, the second token is at start_pos = 1, and so on.
If we do prefill with prompt which has 4 tokens, then during the decode phase,
start_pos = 4.

Why is start_pos neded?
- Attention should not need to know start_pos. However, to apply causal mask,
we can use is_causal parameter (aten API for SDPA is thinking of getting rid
of it). However, the current handling of is_causal assumes that start_pos = 0.
Meaning when we have a query during decode at start_pos = 4, it will be a
single vector of [1, head_dim] for a given head. Key param, derived from kv
cache, will be of size [start_pos + 1, head_dim]. That is all the past tokens
contained in kv cache. If we apply causal mask naively, then the query is
assumed to be at start_pos = 0, and thus all the future tokens (indices 1...4)
in q @ k.T = [1, start_pos], will be masked out for attention calculation.
However, that is not right. Since query is at pos 4, that is 4th token, it
should attend to all previous tokens in the cache. That is 0...start_pos. Thus
we need to pass start_pos.

Can we use attn_mask?
- Yes. Attention mask can be used for the same, however, at the moment attention
mask for our llama model is a boolean mask which requires conversion to -inf for
masked out section. This requires change that may have perf implication, however
we havent really validated this. It is possible that there is no perf
implication. If the mask was float mask, thing will work out-of-the-box. In our
llama definition each layer is storying mask and if we move to float mask, that
can increase memory footprint, which is right now optimized away since
sdpa_with_kv_cache does not use attn_mask.

TODO: Just handle conversion of bool mask to float
*/
/**
 * @brief Implements Flash Attention algorithm on CPU
 *
 * This function computes scaled dot-product attention with optimizations for
 CPU.
 * It supports both regular and quantized attention computation.
 *
 * @tparam scalar_t The data type for computation (e.g., float)
 * @tparam q_split_size Block size for query matrix in tiling algorithm
 * @tparam kv_split_size Block size for key/value matrices in tiling algorithm
 *
 * @param output Output tensor to store attention results
 * @param query Query tensor [Batch x Num_heads x Q_seq_len x Dim_per_head]
 * @param key Key tensor [Batch x Num_heads_kv x KV_seq_len x Dim_per_head]
 * @param value Value tensor [Batch x Num_heads_kv x KV_seq_len x Dim_per_head]
 * @param dropout_p Dropout probability (not used in current implementation)
 * @param is_causal Whether to apply causal mask (lower triangular)
 * @param attn_mask Optional explicit attention mask
 * @param scale Optional custom scaling factor (default: 1/sqrt(head_dim))
 * @param q_zero_points Optional zero points for quantized query
 * @param q_scales Optional scales for quantized query
 * @param k_zero_points Optional zero points for quantized key
 * @param k_scales Optional scales for quantized key
 * @param v_zero_points Optional zero points for quantized value
 * @param v_scales Optional scales for quantized value
 * @param seq_dim Which dimension is sequence dimension.
 If SeqDim::One, then query, key, value are
 expected to be in shape [Batch x Q_seq_len x Dim_per_head x Num_heads] and
 output is expected to be in shape [Batch x Q_seq_len x Dim_per_head x
 Num_heads]
 * @param start_pos Starting position for causal masking in generation
 * @param num_keys_for_causal_attention Number of keys to consider for causal
 attention (-1 for all)
 */
template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    const std::optional<double>& scale,
    const std::optional<Tensor>& q_zero_points,
    const std::optional<Tensor>& q_scales,
    const std::optional<Tensor>& k_zero_points,
    const std::optional<Tensor>& k_scales,
    const std::optional<Tensor>& v_zero_points,
    const std::optional<Tensor>& v_scales,
    const SeqDim seq_dim = SeqDim::TWO,
    const int64_t start_pos = 0,
    const int64_t num_keys_for_causal_attention = -1) {
  (void)dropout_p;

  // Without this we have out-of-bounds writes for
  // causal masking
  static_assert(
      kv_split_size > q_split_size,
      "KV_split_size must be greater than q_split_size");

  constexpr bool is_reduced_type =
      ::executorch::runtime::is_reduced_floating_point_v<scalar_t>;

  ET_CHECK_MSG(
      !is_reduced_type, "FlashAttention does not support reduced types.");
  // Figure out mixed precision a little later
  // using accum_t = at::opmath_type<scalar_t>;
  using accum_t = scalar_t;
  using Vec = vec::Vectorized<accum_t>;
  accum_t scaling_factor = static_cast<accum_t>(calculate_scale(query, scale));

  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t qSize = query.size(2);
  int64_t headSize = query.size(3);
  int64_t kvSize = value.size(2);
  int64_t num_heads_kv = key.size(1);

  if (seq_dim == SeqDim::ONE) {
    num_head = query.size(2);
    num_heads_kv = key.size(2);
    qSize = query.size(1);
    kvSize = value.size(1);
  }

  if (num_keys_for_causal_attention > 0) {
    ET_CHECK_MSG(
        num_keys_for_causal_attention <= kvSize,
        "num_keys_for_causal_attention must be <= kvSize");
    kvSize = num_keys_for_causal_attention;
  }

  ET_CHECK_MSG(
      num_heads_kv <= num_head,
      "FlashAttention does not support num kv heads > num query heads.Got num query heads=%" PRId64
      " num key heads:%" PRId64,
      num_head,
      num_heads_kv);
  ET_CHECK_MSG(
      num_head % num_heads_kv == 0,
      "FlashAttention: num qyery heads must be divisible by num kv heads but got num query heads=%" PRId64
      " and num kv heads=%" PRId64,
      num_head,
      num_heads_kv);
  int64_t num_reps = num_head / num_heads_kv;

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    /*
    TODO: fix this for upcasting attn mask
    if (is_reduced_type) {
      // SHould not come here for now.
      attn_mask.value() = attn_mask.value().to(at::kFloat);
    }
    */
    ET_CHECK_MSG(attn_mask.value().dim() == 2, "attn_mask must be 2D");
    ET_CHECK_MSG(
        attn_mask.value().size(0) == qSize,
        "attn_mask shape mismatch"
        "attn_mask.size(0)=%zd qSize=%" PRId64,
        attn_mask.value().size(0),
        qSize);
    ET_CHECK_MSG(
        attn_mask.value().size(1) == kvSize,
        "attn_mask shape mismatch"
        "attn_mask.size(1)=%zd kvSize=%" PRId64,
        attn_mask.value().size(1),
        kvSize);
  }

  bool is_quantized_sdpa = false;
  is_quantized_sdpa = query.scalar_type() == ScalarType::Char;

  auto strides = query.strides();
  int64_t qStrideB = strides[0];
  int64_t qStrideH = strides[1];
  int64_t qStrideM = strides[2];

  if (seq_dim == SeqDim::ONE) {
    qStrideH = strides[2];
    qStrideM = strides[1];
  }

  strides = key.strides();
  int64_t kStrideB = strides[0];
  int64_t kStrideH = strides[1];
  int64_t kStrideN = strides[2];

  if (seq_dim == SeqDim::ONE) {
    kStrideH = strides[2];
    kStrideN = strides[1];
  }

  strides = value.strides();
  int64_t vStrideB = strides[0];
  int64_t vStrideH = strides[1];
  int64_t vStrideN = strides[2];

  if (seq_dim == SeqDim::ONE) {
    vStrideH = strides[2];
    vStrideN = strides[1];
  }

  int64_t q_quant_params_StrideB = 0;
  int64_t q_quant_params_StrideH = 0;
  int64_t q_quant_params_StrideM = 0;
  int64_t k_quant_params_StrideB = 0;
  int64_t k_quant_params_StrideH = 0;
  int64_t k_quant_params_StrideN = 0;
  int64_t v_quant_params_StrideB = 0;
  int64_t v_quant_params_StrideH = 0;
  int64_t v_quant_params_StrideN = 0;

  if (is_quantized_sdpa) {
    auto q_strides = q_zero_points.value().strides();
    q_quant_params_StrideB = q_strides[0];
    q_quant_params_StrideH = q_strides[1];
    q_quant_params_StrideM = q_strides[2];

    auto k_strides = k_zero_points.value().strides();
    k_quant_params_StrideB = k_strides[0];
    k_quant_params_StrideH = k_strides[1];
    k_quant_params_StrideN = k_strides[2];

    auto v_strides = v_zero_points.value().strides();
    v_quant_params_StrideB = v_strides[0];
    v_quant_params_StrideH = v_strides[1];
    v_quant_params_StrideN = v_strides[2];

    ET_CHECK_MSG(
        (v_quant_params_StrideN == k_quant_params_StrideN) &&
            (v_quant_params_StrideN == q_quant_params_StrideM),
        "Quant params strides must be same for seq dim");

    if (seq_dim == SeqDim::ONE) {
      q_quant_params_StrideH = q_strides[2];
      q_quant_params_StrideM = q_strides[1];

      k_quant_params_StrideH = k_strides[2];
      k_quant_params_StrideN = k_strides[1];

      v_quant_params_StrideH = v_strides[2];
      v_quant_params_StrideN = v_strides[1];
    }
  }

  strides = output.strides();
  int64_t oStrideB = strides[0];
  int64_t oStrideH = strides[1];
  int64_t oStrideM = strides[2];

  if (seq_dim == SeqDim::ONE) {
    oStrideH = strides[2];
    oStrideM = strides[1];
  }

  int64_t mStrideB = 0;
  int64_t mStrideH = 0;
  int64_t mStrideM = 0;
  if (has_attn_mask) {
    // int64_t mStrideB = 0;
    //(has_attn_mask && attn_mask.value().size(0) > 1)
    //    ? attn_mask.value().stride(0)
    //    : 0;
    // int64_t mStrideH = 0;
    //(has_attn_mask && attn_mask.value().size(1) > 1)
    //    ? attn_mask.value().stride(1)
    //    : 0;
    strides = attn_mask.value().strides();
    mStrideM = strides[0];
  }

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
#ifdef ET_USE_THREADPOOL
  int64_t num_thread =
      ::executorch::extension::threadpool::get_threadpool()->get_thread_count();
#else
  int64_t num_thread = 1;
#endif

  // const auto dtype = query.scalar_type();
  // Following will be revisited in the future
  // const auto accumulate_dtype = dtype; // toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  // Since all intermediate compute is accum_t, we need to
  // allocate a buffer accordingly.
  int64_t size_of_intermediate_precision = sizeof(accum_t);
  int64_t size_bytes = size_per_thread * num_thread * query.element_size() *
      size_of_intermediate_precision;
  std::vector<char> buf_vec(size_bytes);
  void* buf = reinterpret_cast<void*>(buf_vec.data());
  // Need to double check the following
  size_bytes = num_thread * qSplitSize * kvSplitSize * query.element_size();
  std::vector<char> buf_reduced_vec(size_bytes);
  void* buf_reduced = reinterpret_cast<void*>(buf_reduced_vec.data());
  // at::Tensor buf_reduced = at::empty(
  //    {num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0},
  //    query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const accum_t* mask_data =
      has_attn_mask ? attn_mask.value().const_data_ptr<accum_t>() : nullptr;
  scalar_t* out_data = output.mutable_data_ptr<scalar_t>();
  accum_t* buf_data = reinterpret_cast<accum_t*>(buf);
  scalar_t* buf_reduced_data =
      is_reduced_type ? reinterpret_cast<scalar_t*>(buf_reduced) : nullptr;

  auto compute_lambda = [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = torch::executor::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    scalar_t* qk_reduced_data = is_reduced_type
        ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize
        : nullptr;

    for (int64_t z = begin; z < end; z++) {
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(
          qk_max_data, -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      // Original flash sdpa wasnt really meant to be used
      // for decode the way we are using via start_pos here.
      // Thus when num_keys is 1 during decode phase, we
      // still need to iterate through all the kv_splits
      // Take start_pos = 130 and k_split_size = 128
      // Here we have to produce [1x130] of q @ k.T
      // when seq_len = 1
      // But if num_keys = 1 then we dont really loop over
      // all kv_splits.
      // When k_split_size > 130, this is not an issue because
      // there is only one iteration of the following loop anyway.
      // Outside of determining how many loop iterations are needed
      // num_keys participates only in causal attention.
      // Rest of the calculation of q @ k.T and @ v.T is same.
      // We dont run into this bug when k_split_size < start_pos + seqlen
      // since there is only one iteration and that applies
      // causal attention correctly.
      // Howeve when k_split_size > start_pos + seqlen, we have
      // more than one iteration, however if we dont adjust num_keys
      // we dont get more than one iteration
      // This is unique to this deployment of flash attention since
      // original implementation wasnt deployed on this way.

      // Some of these bugs can be resolved by relying on attention mask
      // but that requires storing attention mask in float as the current
      // code doesnt support bool attention mask.
      // However, lets just fix that as well.
      int64_t num_keys =
          is_causal ? std::min(m + start_pos + qBlockSize, kvSize) : kvSize;
      int64_t m_start_pos = m + start_pos;
      auto j_kv = j / num_reps;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        // Calculate scale * q @ k.T
        fill_stub(qk_data, static_cast<accum_t>(0), qSplitSize * kvSplitSize);

        const void* q_sub_matrix_data_ptr;
        const void* k_sub_matrix_data_ptr;
        const float* q_scales_ptr = nullptr;
        const float* k_scales_ptr = nullptr;
        const int8_t* q_zero_points_ptr = nullptr;
        const int8_t* k_zero_points_ptr = nullptr;
        int64_t q_offset = i * qStrideB + j * qStrideH + m * qStrideM;
        int64_t k_offset = i * kStrideB + j_kv * kStrideH + n * kStrideN;
        if (is_quantized_sdpa) {
          int64_t q_quant_params_offset = i * q_quant_params_StrideB +
              j * q_quant_params_StrideH + m * q_quant_params_StrideM;
          int64_t k_quant_params_offset = i * k_quant_params_StrideB +
              j_kv * k_quant_params_StrideH + n * k_quant_params_StrideN;
          q_scales_ptr =
              q_scales.value().const_data_ptr<float>() + q_quant_params_offset;
          k_scales_ptr =
              k_scales.value().const_data_ptr<float>() + k_quant_params_offset;
          q_zero_points_ptr = q_zero_points.value().const_data_ptr<int8_t>() +
              q_quant_params_offset;
          k_zero_points_ptr = k_zero_points.value().const_data_ptr<int8_t>() +
              k_quant_params_offset;
          q_sub_matrix_data_ptr = (const int8_t*)(q_data) + q_offset;
          k_sub_matrix_data_ptr = (const int8_t*)(k_data) + k_offset;
        } else {
          q_sub_matrix_data_ptr = (const scalar_t*)(q_data) + q_offset;
          k_sub_matrix_data_ptr = (const scalar_t*)(k_data) + k_offset;
        }
        MaybeQuantizedMatrixData q_sub_matrix_data = MaybeQuantizedMatrixData(
            static_cast<const void*>(q_sub_matrix_data_ptr),
            q_zero_points_ptr,
            q_scales_ptr,
            qBlockSize,
            headSize,
            q_quant_params_StrideM,
            query.scalar_type());
        MaybeQuantizedMatrixData k_sub_matrix_data = MaybeQuantizedMatrixData(
            static_cast<const void*>(k_sub_matrix_data_ptr),
            k_zero_points_ptr,
            k_scales_ptr,
            kvBlockSize,
            headSize,
            k_quant_params_StrideN,
            key.scalar_type());
        _q_at_k_gemm<accum_t>(
            qBlockSize,
            kvBlockSize,
            headSize,
            q_sub_matrix_data,
            qStrideM,
            k_sub_matrix_data,
            kStrideN,
            qk_data);

        // There are 4 cases that is_causal has to cover to fill
        // not-attendable-position with -inf
        /* 1. Everything is attended to. This happens when m_start_pos > n +
        kvSplitSize e.g m_pos [8:15] and n_pos [0:7]. Since you must attend to
        all previous tokens matrix is full
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
           2. Everything is not attended to. However only some tokens at the
        beginning dont attend to everything. This happens when m_start_pos <= n
        + kvSplitSize but m_start_pos + qBlockSize > n + kvSplitSize m_start_pos
        = 8 qBlockSize = 8 n = 4 kvSplitSize = 8 For example m_pos [8:15] but
        n_pos is [4:11]
        + + + + + - - -
        + + + + + + - -
        + + + + + + + -
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
        + + + + + + + +
           3. In this case only last few tokens have something to attend to.
        This happens when m_start_pos < n and m_start_pos + qBlockSize >= n and
        m_start_pos + qBlockSize <= n + kvSplitSize m_start_pos = 8 qBlockSize =
        8 n = 13 kvSplitSize = 8 For example m_pos [8:15] but n_pos is [13:20]
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        + - - - - - - -
        + + - - - - - -
        + + + - - - - -
           4. In this no tokens attend to anything, but we dont really have to
        take care of this case because the loop for (int64_t n = 0; n <
        num_keys; n += kvSplitSize) will exit before that.
        */
        if (is_causal && m_start_pos <= n + kvSplitSize) {
          // For this fn to work k_split_size > q_split_size
          for (int32_t row = 0;
               row < qBlockSize && (m_start_pos + row < n + (kvSplitSize - 1));
               ++row) {
            // When last_col is 0, it means that the entire row is not attended
            // to because m_pos is smaller than n_pos. So everything in n is for
            // future.
            int64_t last_col =
                n > (m_start_pos + row) ? 0 : row + m_start_pos + 1 - n;
            accum_t* row_ptr = qk_data + row * kvBlockSize;
            fill_stub(
                row_ptr + last_col,
                -std::numeric_limits<accum_t>::infinity(),
                kvBlockSize - last_col);
          }
        }
        // Update attention weights with attention mask
        // And apply scaling factor
        // qk <- qk * scaling + attn_mask
        if (has_attn_mask) {
          for (int64_t row = 0; row < qBlockSize; ++row) {
            vec::map2<accum_t>(
                [scaling_factor](Vec x, Vec y) {
                  return x * Vec(scaling_factor) + y;
                },
                qk_data + row * kvBlockSize,
                qk_data + row * kvBlockSize,
                mask_data + i * mStrideB + j * mStrideH + (m + row) * mStrideM +
                    n,
                kvBlockSize);
          }
        }
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < qBlockSize; ++row) {
          if (has_attn_mask) {
            // max per row
            tmp_max = vec::reduce_all<accum_t>(
                [](Vec& x, Vec& y) { return vec::maximum(x, y); },
                qk_data + row * kvBlockSize,
                kvBlockSize);
          } else {
            // apply scaling factor and max per row in fusion
            _mul_reduce_max_fusion_kernel(
                qk_data + row * kvBlockSize,
                scaling_factor,
                kvBlockSize,
                qk_data + row * kvBlockSize,
                tmp_max);
          }
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
            // to avoid `nan = exp2f(-inf - (-inf))`
            fill_stub(
                conditional_data_ptr(qk_data, qk_reduced_data) +
                    row * kvBlockSize,
                static_cast<scalar_t>(0),
                kvBlockSize);
          } else {
            // qk <- exp(qk - max) and sum per row
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(
                qk_data + row * kvBlockSize,
                kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) +
                    row * kvBlockSize,
                tmp_sum);
            // exp_tmp <- exp(max[row] - max)
            exp_tmp = std::exp(qk_max_data[row] - tmp_max);
            // sum[row] <- sum + exp_tmp * sum[row]
            qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
            // max[row] <- max
            qk_max_data[row] = tmp_max;
            // dst <- dst * exp_tmp
            if (n > 0) {
              vec::map<accum_t>(
                  [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                  dst_data + row * headSize,
                  dst_data + row * headSize,
                  headSize);
            }
          }
        }

        const void* v_sub_matrix_data_ptr;
        const float* v_scales_ptr = nullptr;
        const int8_t* v_zero_points_ptr = nullptr;
        int64_t v_offset = i * vStrideB + j_kv * vStrideH + n * vStrideN;
        if (is_quantized_sdpa) {
          int64_t v_quant_params_offset = i * v_quant_params_StrideB +
              j_kv * v_quant_params_StrideH + n * v_quant_params_StrideN;
          v_scales_ptr =
              v_scales.value().const_data_ptr<float>() + v_quant_params_offset;
          v_zero_points_ptr = v_zero_points.value().const_data_ptr<int8_t>() +
              v_quant_params_offset;
          v_sub_matrix_data_ptr = (const int8_t*)(v_data) + v_offset;
        } else {
          v_sub_matrix_data_ptr = (const scalar_t*)(v_data) + v_offset;
        }
        MaybeQuantizedMatrixData v_sub_matrix_data = MaybeQuantizedMatrixData(
            static_cast<const void*>(v_sub_matrix_data_ptr),
            v_zero_points_ptr,
            v_scales_ptr,
            kvBlockSize,
            headSize,
            v_quant_params_StrideN,
            value.scalar_type());
        // Calculate Softmax(q @ k.T) @ v
        _qk_at_v_gemm<accum_t>(
            qBlockSize,
            headSize,
            kvBlockSize,
            qk_data,
            kvBlockSize,
            v_sub_matrix_data,
            vStrideN,
            dst_data,
            headSize,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1));
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < qBlockSize; ++row) {
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        vec::map<scalar_t>(
            [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
            out_data + i * oStrideB + j * oStrideH + m * oStrideM +
                row * oStrideM,
            dst_data + row * headSize,
            headSize);
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  };
  torch::executor::parallel_for(
      0, batchSize * num_head * qSlice, 1, compute_lambda);
}
} // namespace sdpa::impl
} // namespace native
} // namespace executor
} // namespace torch
