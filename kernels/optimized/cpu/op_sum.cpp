/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstring>
#include <optional>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;

// Forward decl of the portable kernel — used as a fallback for shapes and
// dtype combinations the optimized path doesn't specialize. Both libraries
// live in the same binary, so direct call is fine.
Tensor& sum_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    std::optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    std::optional<ScalarType> dtype,
    Tensor& out);

namespace {

// Contiguous innermost reduction: sum each row of the inner axis into one
// scalar. fp16/bf16 accumulate in fp32 for precision; fp32 accumulates in
// fp32 directly. Uses at::vec::Vectorized for cross-arch SIMD.
template <typename CTYPE>
inline void sum_innermost(
    const CTYPE* in,
    CTYPE* out,
    int64_t outer_size,
    int64_t reduce_size) {
  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = static_cast<int64_t>(Vec::size());
  for (int64_t i = 0; i < outer_size; ++i) {
    const CTYPE* row = in + i * reduce_size;
    Vec acc(0.0f);
    int64_t j = 0;
    for (; j + kVecSize - 1 < reduce_size; j += kVecSize) {
      if constexpr (std::is_same_v<CTYPE, float>) {
        acc = acc + Vec::loadu(row + j);
      } else {
        // Half / BFloat16: load N elements, convert to float, add.
        float tmp[kVecSize];
        for (int64_t k = 0; k < kVecSize; ++k) {
          tmp[k] = static_cast<float>(row[j + k]);
        }
        acc = acc + Vec::loadu(tmp);
      }
    }
    float sum = at::vec::vec_reduce_all<float>(
        [](Vec a, Vec b) { return a + b; }, acc);
    for (; j < reduce_size; ++j) {
      sum += static_cast<float>(row[j]);
    }
    out[i] = static_cast<CTYPE>(sum);
  }
}

// Non-innermost (strided) single-dim reduction. For each (outer, inner) pair,
// sum over reduce_size elements spaced `inner_size` apart. Vectorize across
// the contiguous inner axis (so each add-step processes kVecSize output
// positions at once).
template <typename CTYPE>
inline void sum_strided(
    const CTYPE* in,
    CTYPE* out,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size) {
  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = static_cast<int64_t>(Vec::size());
  const int64_t outer_stride = reduce_size * inner_size;
  for (int64_t o = 0; o < outer_size; ++o) {
    const CTYPE* in_o = in + o * outer_stride;
    CTYPE* out_o = out + o * inner_size;
    int64_t j = 0;
    for (; j + kVecSize - 1 < inner_size; j += kVecSize) {
      Vec acc(0.0f);
      for (int64_t k = 0; k < reduce_size; ++k) {
        const CTYPE* p = in_o + k * inner_size + j;
        if constexpr (std::is_same_v<CTYPE, float>) {
          acc = acc + Vec::loadu(p);
        } else {
          float tmp[kVecSize];
          for (int64_t m = 0; m < kVecSize; ++m) {
            tmp[m] = static_cast<float>(p[m]);
          }
          acc = acc + Vec::loadu(tmp);
        }
      }
      if constexpr (std::is_same_v<CTYPE, float>) {
        acc.store(out_o + j);
      } else {
        float tmp[kVecSize];
        acc.store(tmp);
        for (int64_t m = 0; m < kVecSize; ++m) {
          out_o[j + m] = static_cast<CTYPE>(tmp[m]);
        }
      }
    }
    for (; j < inner_size; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < reduce_size; ++k) {
        sum += static_cast<float>(in_o[k * inner_size + j]);
      }
      out_o[j] = static_cast<CTYPE>(sum);
    }
  }
}

} // namespace

Tensor& opt_sum_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    std::optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    std::optional<ScalarType> dtype,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (in.numel() == 0) {
    if (out.numel() > 0) {
      std::memset(out.mutable_data_ptr(), 0, out.nbytes());
    }
    return out;
  }

  // Fast path: single reduction dim, matching dtype, non-complex, contiguous.
  // Anything else falls through to the portable kernel.
  const bool fast_eligible = dim_list.has_value() &&
      dim_list.value().size() == 1 &&
      in.scalar_type() == out.scalar_type() &&
      !executorch::runtime::isComplexType(in.scalar_type()) &&
      tensor_is_contiguous(in);

  if (fast_eligible) {
    const int64_t d = dim_list.value()[0] < 0 ? dim_list.value()[0] + in.dim()
                                              : dim_list.value()[0];
    int64_t outer_size = 1, reduce_size = in.size(d), inner_size = 1;
    for (int64_t i = 0; i < d; ++i) {
      outer_size *= in.size(i);
    }
    for (int64_t i = d + 1; i < in.dim(); ++i) {
      inner_size *= in.size(i);
    }

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "sum.IntList_out";
    bool handled = false;
    ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
      const CTYPE* ip = in.const_data_ptr<CTYPE>();
      CTYPE* op = out.mutable_data_ptr<CTYPE>();
      if (inner_size == 1) {
        sum_innermost<CTYPE>(ip, op, outer_size, reduce_size);
        handled = true;
      } else {
        sum_strided<CTYPE>(ip, op, outer_size, reduce_size, inner_size);
        handled = true;
      }
    });
    if (handled) {
      return out;
    }
  }

  // Fallback.
  return sum_dim_out(ctx, in, dim_list, keepdim, dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
