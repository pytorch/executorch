/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_topk_args(
    const Tensor& in,
    int64_t k,
    int64_t dim,
    Tensor& values,
    Tensor& indices) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, values));
  ET_LOG_AND_RETURN_IF_FALSE(indices.scalar_type() == ScalarType::Long);
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  if (dim < 0) {
    dim += nonzero_dim(in);
  }
  ET_CHECK_OR_RETURN_FALSE(
      k >= 0 && k <= nonempty_size(in, dim),
      "selected index k out of range; k = %" PRId64 ", dim = %" PRId64
      ", in.dim() = %" ET_PRI_TENSOR_DIM ", nonempty_size(in, dim) = %zd",
      k,
      dim,
      in.dim(),
      nonempty_size(in, dim));
  return true;
}

bool get_topk_target_size(
    const Tensor& in,
    int64_t k,
    int64_t dim,
    Tensor::SizesType* target_size,
    size_t* target_dim) {
  *target_dim = in.dim();
  for (const auto i : c10::irange(*target_dim)) {
    if (static_cast<int64_t>(i) == dim) {
      target_size[i] = k;
    } else {
      target_size[i] = in.size(i);
    }
  }
  return true;
}

template <typename T>
bool float_less_than(T x, T y) {
  if constexpr (std::is_integral_v<T>) {
    return x < y;
  }
  return (!utils::isnan_override(x) && utils::isnan_override(y)) || x < y;
}

template <typename CTYPE, typename elem_t = std::pair<CTYPE, int64_t>>
void perform_topk(
    const Tensor& in,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices,
    elem_t* queue) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* values_data = values.mutable_data_ptr<CTYPE>();
  long* indices_data = indices.mutable_data_ptr<long>();

  if (in.dim() == 0) {
    values_data[0] = in_data[0];
    indices_data[0] = 0;
    return;
  }

  if (k == 0) {
    return;
  }

  const size_t outer_size = getLeadingDims(in, dim);

  const size_t dim_size = in.size(dim);
  const size_t dim_stride = in.strides()[dim];

  const size_t outer_stride_in = dim_size * dim_stride;
  const size_t outer_stride_out = k * dim_stride;

  bool use_partial_sort = k * 64 <= static_cast<int64_t>(dim_size);

  // Loop through all outer dimensions
  for (const auto outer_idx : c10::irange(outer_size)) {
    size_t outer_in = outer_idx * outer_stride_in;
    size_t outer_out = outer_idx * outer_stride_out;
    // Loop through all inner dimensions
    for (const auto inner_idx : c10::irange(dim_stride)) {
      size_t base_in = outer_in + inner_idx;
      size_t base_out = outer_out + inner_idx;

      // Populate the queue with the values from the input tensor
      for (const auto i : c10::irange(dim_size)) {
        size_t in_ix = base_in + i * dim_stride;
        queue[i].first = in_data[in_ix];
        queue[i].second = i;
      }

      // Perform topk on the queue
      const auto elem_greater = [](const elem_t& x, const elem_t& y) -> bool {
        return float_less_than(y.first, x.first);
      };
      const auto elem_less = [](const elem_t& x, const elem_t& y) -> bool {
        return float_less_than(x.first, y.first);
      };
      const auto cmp = largest ? elem_greater : elem_less;
      if (use_partial_sort) {
        std::partial_sort(queue, queue + k, queue + dim_size, cmp);
      } else {
        std::nth_element(queue, queue + k - 1, queue + dim_size, cmp);
        if (sorted) {
          std::sort(queue, queue + k - 1, cmp);
        }
      }

      // Write the topk values and indices to the output tensors
      for (const auto i : c10::irange(k)) {
        size_t out_ix = base_out + i * dim_stride;

        values_data[out_ix] = queue[i].first;
        indices_data[out_ix] = queue[i].second;
      }
    }
  }
}

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size) {
  Result<void*> temp_mem_res = ctx.allocate_temp(size);
  return temp_mem_res.ok() ? temp_mem_res.get() : nullptr;
}

} // namespace

std::tuple<Tensor&, Tensor&> topk_values(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  auto out = std::tuple<Tensor&, Tensor&>({values, indices});

  ET_KERNEL_CHECK(
      ctx, check_topk_args(in, k, dim, values, indices), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType target_size[kTensorDimensionLimit];
  size_t target_dim = 0;
  get_topk_target_size(in, k, dim, target_size, &target_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(values, {target_size, target_dim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(indices, {target_size, target_dim}) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "topk.values";

  if (in.numel() == 0 || (k == 0 && in.dim() > 0)) {
    return out;
  }

  bool temp_mem_allocated = false;

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    using elem_t = std::pair<CTYPE, int64_t>;
    size_t temp_mem_size = nonempty_size(in, dim) * sizeof(elem_t);

    elem_t* queue = (elem_t*)allocate_temp_memory(ctx, temp_mem_size);
    if (queue == nullptr) {
      return;
    }
    temp_mem_allocated = true;

    perform_topk<CTYPE>(in, k, dim, largest, sorted, values, indices, queue);
  });

  ET_KERNEL_CHECK(ctx, temp_mem_allocated, MemoryAllocationFailed, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
