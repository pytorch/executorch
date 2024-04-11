/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/custom_ops/op_sdpa.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <array>
#include <vector>

#ifdef ET_USE_THREADPOOL
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#include <executorch/extension/parallel/thread_parallel.h>
#endif
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace torch {
namespace executor {

namespace native {

namespace util {

constexpr size_t kKVDim = 4;

template <typename T>
inline void _store(T* dst, ::executorch::vec::Vectorized<T> src) {
  src.store(dst);
}

/*
inline void _store(::Half* dst, at::vec::Vectorized<float> src) {
  //fp16_ieee_to_fp32_value
  auto res = at::vec::convert_float_half(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}
*/

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
namespace vec = ::executorch::vec;
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
        torch::executor::is_reduced_floating_point_v<scalar_t>,
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
    bool is_with_kv_cache = false) {
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

  constexpr bool is_reduced_type =
      torch::executor::is_reduced_floating_point_v<scalar_t>;

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

  if (is_with_kv_cache) {
    num_head = query.size(2);
    qSize = query.size(1);
    kvSize = value.size(1);
  }

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

  if (is_with_kv_cache) {
    qStrideH = strides[2];
    qStrideM = strides[1];
  }

  strides = key.strides();
  int64_t kStrideB = strides[0];
  int64_t kStrideH = strides[1];
  int64_t kStrideN = strides[2];

  if (is_with_kv_cache) {
    kStrideH = strides[2];
    kStrideN = strides[1];
  }

  strides = value.strides();
  int64_t vStrideB = strides[0];
  int64_t vStrideH = strides[1];
  int64_t vStrideN = strides[2];

  if (is_with_kv_cache) {
    vStrideH = strides[2];
    vStrideN = strides[1];
  }

  strides = output.strides();
  int64_t oStrideB = strides[0];
  int64_t oStrideH = strides[1];
  int64_t oStrideM = strides[2];

  if (is_with_kv_cache) {
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
      torch::executorch::threadpool::get_threadpool()->get_thread_count();
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
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  accum_t* mask_data =
      has_attn_mask ? attn_mask.value().data_ptr<accum_t>() : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
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
      int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
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
            k_data + i * kStrideB + j * kStrideH + n * kStrideN,
            kStrideN,
            q_data + i * qStrideB + j * qStrideH + m * qStrideM,
            qStrideM,
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);
        // Apply causal mask, fill unused with -inf
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (int32_t row = 0; row < qBlockSize; ++row) {
            int64_t last_col = m + row - n;
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
            v_data + i * vStrideB + j * vStrideH + n * vStrideN,
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
    int64_t seq_length) {
  ET_CHECK_MSG(seq_length == 1, "seq_length must be 1");
  ET_CHECK_MSG(
      projected_value.size(0) == 1,
      "projected_value must have batch size of 1");
  ET_CHECK_MSG(cache.size(0) == 1, "cache must have batch size of 1");
  ET_CHECK_MSG(
      is_contiguous_dim_order(
          projected_value.dim_order().data(), projected_value.dim()),
      "projected value must be in contiguous dim order");
  const void* projected_value_data = projected_value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();

  ET_CHECK_MSG(projected_value_data != nullptr, "projected_value data is null");
  ET_CHECK_MSG(cache_data, "cache data is null");

  auto strides = cache.strides();
  exec_aten::StridesType seq_dim_stride = strides[1];
  exec_aten::SizesType pos_offset = start_pos * seq_dim_stride;
  exec_aten::SizesType pos_offset_bytes =
      pos_offset * projected_value.element_size();
  exec_aten::SizesType num_bytes =
      projected_value.numel() * projected_value.element_size();
  // NOLINTNEXTLINE
  std::memcpy(
      (uint8_t*)cache_data + pos_offset_bytes, projected_value_data, num_bytes);
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
Tensor& sdpa_with_kv_cache_out(
    RuntimeContext& ctx,
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

  auto q_seq_len = q_projected.size(1);

  std::array<exec_aten::DimOrderType, util::kKVDim> sliced_key_dim_order{
      0, 1, 2, 3};
  std::array<exec_aten::SizesType, util::kKVDim> sliced_key_sizes;
  sliced_key_sizes[0] = key_cache.size(0);
  sliced_key_sizes[1] = start_pos + seq_len; // key_cache.size(2);
  sliced_key_sizes[2] = key_cache.size(2);
  sliced_key_sizes[3] = key_cache.size(3);
  std::array<exec_aten::StridesType, util::kKVDim> sliced_key_strides;
  dim_order_to_stride_nocheck(
      sliced_key_sizes.data(),
      sliced_key_dim_order.data(),
      util::kKVDim,
      sliced_key_strides.data());
  void* key_cache_data = key_cache.mutable_data_ptr();
  TensorImpl k_impl = TensorImpl(
      key_cache.scalar_type(),
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
  sliced_value_sizes[0] = value_cache.size(0);
  sliced_value_sizes[1] = start_pos + seq_len; // value_cache.size(2);
  sliced_value_sizes[2] = value_cache.size(2);
  sliced_value_sizes[3] = value_cache.size(3);
  std::array<exec_aten::StridesType, util::kKVDim> sliced_value_strides;
  dim_order_to_stride_nocheck(
      sliced_value_sizes.data(),
      sliced_value_dim_order.data(),
      util::kKVDim,
      sliced_value_strides.data());
  void* value_cache_data = value_cache.mutable_data_ptr();
  TensorImpl value_impl = TensorImpl(
      value_cache.scalar_type(),
      util::kKVDim,
      sliced_value_sizes.data(),
      value_cache_data,
      sliced_value_dim_order.data(),
      sliced_value_strides.data(),
      TensorShapeDynamism::STATIC);
  Tensor sliced_value_cache(&value_impl);

  // Is this true?
  // Cant do this as is because the expectation of this kernel is
  // that q, k, v are [B, num heads, seq length, head dim]
  // and the cache is [B, max seq len, num heads, head dim]
  // and q, k, v are all [B, seq length, num heads, head dim]

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, q_projected.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  // TODO(task): replace the template param selection logic
  // with whatever apprpriately makes more sense for
  ET_SWITCH_FLOAT_TYPES(
      q_projected.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (q_seq_len >= 768) {
          cpu_flash_attention<CTYPE, 256, 512>(
              output,
              q_projected,
              sliced_key_cache,
              sliced_value_cache,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              true);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention<CTYPE, 64, 512>(
              output,
              q_projected,
              sliced_key_cache,
              sliced_value_cache,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              true);
        } else {
          cpu_flash_attention<CTYPE, 32, 512>(
              output,
              q_projected,
              sliced_key_cache,
              sliced_value_cache,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              true);
        }
      });
  return output;
}
} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "sdpa_with_kv_cache.out",
    torch::executor::native::sdpa_with_kv_cache_out);
