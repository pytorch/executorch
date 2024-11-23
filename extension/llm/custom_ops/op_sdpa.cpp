/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_sdpa.h>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <array>
#include <vector>

#ifdef ET_USE_THREADPOOL
#include <executorch/extension/parallel/thread_parallel.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace torch {
namespace executor {

namespace native {

namespace util {

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

inline double calculate_scale(const Tensor& query, optional<double> scale) {
  const auto softmax_scale =
      scale.has_value() ? scale.value() : 1.0 / std::sqrt(query.size(3));
  return softmax_scale;
}

} // namespace util
namespace vec = ::at::vec;
using Tensor = exec_aten::Tensor;

namespace {

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void
_exp_reduce_sum_fusion_kernel(T1* a, const int& size, T2* out, T1& val) {
  auto vec_size = vec::Vectorized<T1>::size();
  auto vec_max = vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = vec::Vectorized<T1>(tmp_sum);
  for (int i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    // Replace with exp_u20 later
    // auto tmp2 = tmp1.exp_u20();
    auto tmp2 = tmp1.exp();
    vec_tmp_sum += tmp2;
    util::_store(out + i, tmp2);
  }
  tmp_sum = vec::vec_reduce_all<T1>(
      [](vec::Vectorized<T1>& x, vec::Vectorized<T1>& y) { return x + y; },
      vec_tmp_sum);
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
    util::_store(out + i, tmp1);
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
template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    const optional<Tensor>& attn_mask,
    const optional<double>& scale,
    bool is_seq_at_dim_1 = false,
    const int64_t start_pos = 0) {
  (void)dropout_p;
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)

  /*
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor key = k.transpose(1, 2);
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor value = v.transpose(1, 2);
  */

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
  accum_t scaling_factor =
      static_cast<accum_t>(util::calculate_scale(query, scale));

  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t qSize = query.size(2);
  int64_t headSize = query.size(3);
  int64_t kvSize = value.size(2);
  int64_t num_heads_kv = key.size(1);

  if (is_seq_at_dim_1) {
    num_head = query.size(2);
    num_heads_kv = key.size(2);
    qSize = query.size(1);
    kvSize = value.size(1);
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
        attn_mask.value().size(0) == qSize, "attn_mask shape mismatch");
    ET_CHECK_MSG(
        attn_mask.value().size(1) == kvSize,
        "attn_mask shape mismatch"
        "attn_mask.size(1)=%zd kvSize=%" PRId64,
        attn_mask.value().size(1),
        kvSize);
  }

  auto strides = query.strides();
  int64_t qStrideB = strides[0];
  int64_t qStrideH = strides[1];
  int64_t qStrideM = strides[2];

  if (is_seq_at_dim_1) {
    qStrideH = strides[2];
    qStrideM = strides[1];
  }

  strides = key.strides();
  int64_t kStrideB = strides[0];
  int64_t kStrideH = strides[1];
  int64_t kStrideN = strides[2];

  if (is_seq_at_dim_1) {
    kStrideH = strides[2];
    kStrideN = strides[1];
  }

  strides = value.strides();
  int64_t vStrideB = strides[0];
  int64_t vStrideH = strides[1];
  int64_t vStrideN = strides[2];

  if (is_seq_at_dim_1) {
    vStrideH = strides[2];
    vStrideN = strides[1];
  }

  strides = output.strides();
  int64_t oStrideB = strides[0];
  int64_t oStrideH = strides[1];
  int64_t oStrideM = strides[2];

  if (is_seq_at_dim_1) {
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

  int64_t size_bytes = size_per_thread * num_thread * query.element_size();
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
    util::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
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
      auto j_kv = j / num_reps;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        // Calculate scale * q @ k.T
        fill_stub(qk_data, static_cast<accum_t>(0), qSplitSize * kvSplitSize);
        ::executorch::cpublas::gemm(
            ::executorch::cpublas::TransposeType::Transpose,
            ::executorch::cpublas::TransposeType::NoTranspose,
            kvBlockSize,
            qBlockSize,
            headSize,
            static_cast<accum_t>(1),
            k_data + i * kStrideB + j_kv * kStrideH + n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH + m * qStrideM,
            qStrideM,
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);
        // Apply causal mask, fill unused, i.e. future values, with -inf
        // Say you have q @ k.T size = [16, 32]
        // With qblock size = 4, say you are processing
        // q seq len dim = 8:11.
        // Say kvSplitSize = 4
        // Then for causal mask, the entries that needs to be
        // ignored are
        // [8, 9:31], [9, 10:31], [10, 10:31], [11, 11:31]
        // Following condition says that num_keys = 8 + 4 =12
        // (num_keys - n) <= kvSplitSize
        // num_keys <= n + kvSplitSize
        // If n + kvSplitSize is larger than 12, then some
        // entries need masked out. In our example n = 4
        // will qualify for that
        if (is_causal && num_keys - n <= kvSplitSize) {
          // For this fn to work k_split_size > q_split_size
          for (int32_t row = 0; row < qBlockSize; ++row) {
            int64_t last_col = m + (row + start_pos) - n;
            accum_t* row_ptr = qk_data + row * kvBlockSize;
            fill_stub(
                row_ptr + last_col + 1,
                -std::numeric_limits<accum_t>::infinity(),
                kvBlockSize - last_col - 1);
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
        // Calculate Softmax(q @ k.T) @ v
        ::executorch::cpublas::gemm(
            ::executorch::cpublas::TransposeType::NoTranspose,
            ::executorch::cpublas::TransposeType::NoTranspose,
            headSize,
            qBlockSize,
            kvBlockSize,
            static_cast<accum_t>(1),
            v_data + i * vStrideB + j_kv * vStrideH + n * vStrideN,
            vStrideN,
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);
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
      util::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  };
  torch::executor::parallel_for(
      0, batchSize * num_head * qSlice, 1, compute_lambda);
}

bool validate_flash_attention_args(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const optional<Tensor>& attn_mask) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(query.dim() == 4, "query must be a 4D tensor");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(key.dim() == 4, "key must be a 4D tensor");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(value.dim() == 4, "value must be a 4D tensor");

  // Sizes
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (query.scalar_type() == ScalarType::Float), "Query must be Float type");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (query.scalar_type() == key.scalar_type()) &&
          (query.scalar_type() == value.scalar_type()),
      "Key and Value must have the same data type as Query");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !attn_mask.has_value() || attn_mask.value().dim() == 2,
      "Attention mask must be a 2D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !attn_mask.has_value() ||
          attn_mask.value().scalar_type() == query.scalar_type(),
      "Attention mask must be a 2D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(query.dim_order().data(), query.dim()),
      "key cache must be in contiguous dim order");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(key.dim_order().data(), key.dim()),
      "value cache must be in contiguous dim order");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(value.dim_order().data(), value.dim()),
      "value cache must be in contiguous dim order");

  if (attn_mask.has_value()) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        is_contiguous_dim_order(
            attn_mask.value().dim_order().data(), attn_mask.value().dim()),
        "value cache must be in contiguous dim order");
  }

  return true;
}

bool validate_cache_params(
    const Tensor& k_cache,
    const Tensor& v_cache,
    int64_t start_pos,
    int64_t seq_length) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      k_cache.dim() == 4, "kcache must be a 4D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      v_cache.dim() == 4, "v_cache must be a 4D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      start_pos < k_cache.size(1),
      "start_pos must be less than key cache at dim 1");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      start_pos < v_cache.size(1),
      "start_pos must be less than value cache at dim 1");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (start_pos + seq_length) <= k_cache.size(1),
      "start_post + seq_length must be less than max seq length supported by key cache."
      "start pos: %" PRId64 ", seq_length: %" PRId64
      "."
      "key cache size: %zd",
      start_pos,
      seq_length,
      k_cache.size(1));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (start_pos + seq_length) <= v_cache.size(1),
      "start_post + seq_length must be less than max seq length supported by key cache."
      "start pos: %" PRId64 ", seq_length: %" PRId64
      "."
      "value cache size: %zd",
      start_pos,
      seq_length,
      v_cache.size(1));

  // Make sure they are in contiguous dim order
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(k_cache.dim_order().data(), k_cache.dim()),
      "key cache must be in contiguous dim order");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(v_cache.dim_order().data(), v_cache.dim()),
      "value cache must be in contiguous dim order");

  return true;
}

// TODO: seq_length is not yet used for copy
void update_cache(
    const Tensor& projected_value,
    const Tensor& cache,
    int64_t start_pos,
    int64_t seq_length) { // NOLINT: unused parameter 'seq_length'
  // 1) Cache shape should be [bs, max_seq_len, num heads, head dim]
  // 2) projected_value shape should be [bs, seq_len, num heads, head dim]
  // 3) We're updating the cache with projected_value, at position start_pos

  ET_CHECK_MSG(
      projected_value.size(0) == cache.size(0),
      "projected_value batch size should be equal to the cache batch size.");
  ET_CHECK_MSG(
      projected_value.size(2) == cache.size(2),
      "projected_value number of heads should be equal to the cache number of heads.");
  ET_CHECK_MSG(
      projected_value.size(3) == cache.size(3),
      "projected_value embedding dimension should be equal to the cache embedding dimension.");
  ET_CHECK_MSG(
      projected_value.element_size() == cache.element_size(),
      "projected_value data type size should be equal to the cache data type size.");

  ET_CHECK_MSG(
      is_contiguous_dim_order(
          projected_value.dim_order().data(), projected_value.dim()),
      "projected value must be in contiguous dim order");
  const void* projected_value_data = projected_value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();

  ET_CHECK_MSG(projected_value_data != nullptr, "projected_value data is null");
  ET_CHECK_MSG(cache_data, "cache data is null");

  auto cache_strides = cache.strides();
  exec_aten::StridesType cache_batch_dim_stride = cache_strides[0];
  exec_aten::StridesType cache_seq_dim_stride = cache_strides[1];

  auto value_strides = projected_value.strides();
  exec_aten::StridesType value_batch_dim_stride = value_strides[0];

  exec_aten::SizesType num_bytes_to_copy =
      (projected_value.numel() / projected_value.size(0)) *
      projected_value.element_size();

  for (int64_t batch_line = 0; batch_line < projected_value.size(0);
       ++batch_line) {
    exec_aten::SizesType cache_pos_offset =
        (batch_line * cache_batch_dim_stride +
         start_pos * cache_seq_dim_stride) *
        cache.element_size();
    exec_aten::SizesType value_pos_offset =
        (batch_line * value_batch_dim_stride) * cache.element_size();

    std::memcpy(
        (uint8_t*)cache_data + cache_pos_offset,
        (uint8_t*)projected_value_data + value_pos_offset,
        num_bytes_to_copy);
  }
}

} // anonymous namespace

Tensor& flash_attention_kernel_out(
    RuntimeContext& ctx,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  (void)ctx;
  ET_KERNEL_CHECK(
      ctx,
      validate_flash_attention_args(query, key, value, attn_mask),
      InvalidArgument,
      output);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, query.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  auto q_seq_len = query.size(2);

  ET_SWITCH_FLOAT_TYPES(
      query.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (q_seq_len >= 768) {
          cpu_flash_attention<CTYPE, 256, 512>(
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention<CTYPE, 64, 512>(
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale);
        } else {
          cpu_flash_attention<CTYPE, 32, 512>(
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale);
        }
      });
  return output;
}

/*
  Input params
  @param[in] q_projected Projected query with query weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] k_projected Projected query with key weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] v_projected Projected query with value weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous k_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous v_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  ....
  @param[in] start_pos: sequence position
  @param[in] seq_len: Seq length. e.g. seq_len dim of q_projected.
*/
Tensor& custom_sdpa_out(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      !attn_mask.has_value() || !is_causal,
      InvalidArgument,
      output,
      "attn_mask and is_causal cannot be set at the same time");

  ET_CHECK_MSG(q.dim() == 4, "query must be a 4D tensor");

  const int64_t seq_len = q.size(1);
  auto q_seq_len = q.size(1);

  // Refactor the following into create_view util perhaps using
  // TensorPtr
  std::array<exec_aten::DimOrderType, util::kKVDim> sliced_key_dim_order{
      0, 1, 2, 3};
  std::array<exec_aten::SizesType, util::kKVDim> sliced_key_sizes;
  sliced_key_sizes[0] = k.size(0);
  sliced_key_sizes[1] = start_pos + seq_len; // key_cache.size(2);
  sliced_key_sizes[2] = k.size(2);
  sliced_key_sizes[3] = k.size(3);
  std::array<exec_aten::StridesType, util::kKVDim> sliced_key_strides;
  dim_order_to_stride_nocheck(
      sliced_key_sizes.data(),
      sliced_key_dim_order.data(),
      util::kKVDim,
      sliced_key_strides.data());
  // since the cache is sliced, the batch stride needs to stay the same.
  sliced_key_strides[0] = k.strides()[0];
  void* key_cache_data = k.mutable_data_ptr();
  TensorImpl k_impl = TensorImpl(
      k.scalar_type(),
      util::kKVDim,
      sliced_key_sizes.data(),
      key_cache_data,
      sliced_key_dim_order.data(),
      sliced_key_strides.data(),
      TensorShapeDynamism::STATIC);
  Tensor sliced_key_cache(&k_impl);

  std::array<exec_aten::DimOrderType, util::kKVDim> sliced_value_dim_order{
      0, 1, 2, 3};
  std::array<exec_aten::SizesType, util::kKVDim> sliced_value_sizes;
  sliced_value_sizes[0] = v.size(0);
  sliced_value_sizes[1] = start_pos + seq_len; // value_cache.size(2);
  sliced_value_sizes[2] = v.size(2);
  sliced_value_sizes[3] = v.size(3);
  std::array<exec_aten::StridesType, util::kKVDim> sliced_value_strides;
  dim_order_to_stride_nocheck(
      sliced_value_sizes.data(),
      sliced_value_dim_order.data(),
      util::kKVDim,
      sliced_value_strides.data());
  // since the cache is sliced, the batch stride needs to stay the same.
  sliced_value_strides[0] = v.strides()[0];
  void* value_cache_data = v.mutable_data_ptr();
  TensorImpl value_impl = TensorImpl(
      v.scalar_type(),
      util::kKVDim,
      sliced_value_sizes.data(),
      value_cache_data,
      sliced_value_dim_order.data(),
      sliced_value_strides.data(),
      TensorShapeDynamism::STATIC);
  Tensor sliced_value_cache(&value_impl);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, q.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  // TODO(task): replace the template param selection logic
  // with whatever apprpriately makes more sense for
  ET_SWITCH_FLOAT_TYPES(q.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
    // TODO we need to re-evaluate this for ARM CPUs
    // And there can be many so instead of templatizing
    // we might consider another appraoch
    if (q_seq_len >= 768) {
      cpu_flash_attention<CTYPE, 256, 512>(
          output,
          q,
          sliced_key_cache,
          sliced_value_cache,
          dropout_p,
          is_causal,
          attn_mask,
          scale,
          true, /* is_seq_at_dim_1 */
          start_pos);
    } else if (q_seq_len >= 192) {
      cpu_flash_attention<CTYPE, 64, 512>(
          output,
          q,
          sliced_key_cache,
          sliced_value_cache,
          dropout_p,
          is_causal,
          attn_mask,
          scale,
          true, /* is_seq_at_dim_1 */
          start_pos);
    } else {
      cpu_flash_attention<CTYPE, 32, 512>(
          output,
          q,
          sliced_key_cache,
          sliced_value_cache,
          dropout_p,
          is_causal,
          attn_mask,
          scale,
          true, /* is_seq_at_dim_1 */
          start_pos);
    }
  });
  return output;
}
/*
  Input params
  @param[in] q_projected Projected query with query weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] k_projected Projected query with key weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] v_projected Projected query with value weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous k_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous v_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  ....
  @param[in] start_pos: sequence position
  @param[in] seq_len: Seq length. e.g. seq_len dim of q_projected.
*/
Tensor& sdpa_with_kv_cache_out(
    KernelRuntimeContext& ctx,
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  (void)ctx;
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(key_cache, value_cache, start_pos, seq_len),
      InvalidArgument,
      output);

  ET_CHECK_MSG(q_projected.dim() == 4, "query must be a 4D tensor");

  update_cache(k_projected, key_cache, start_pos, seq_len);
  update_cache(v_projected, value_cache, start_pos, seq_len);

  custom_sdpa_out(
      ctx,
      q_projected,
      key_cache,
      value_cache,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);

  return output;
}
} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "sdpa_with_kv_cache.out",
    torch::executor::native::sdpa_with_kv_cache_out);

EXECUTORCH_LIBRARY(
    llama,
    "custom_sdpa.out",
    torch::executor::native::custom_sdpa_out);
