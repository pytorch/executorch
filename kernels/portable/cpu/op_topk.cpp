/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>

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
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      k >= 0 && k <= nonempty_size(in, dim), "selected index k out of range");
  return true;
}

bool get_topk_target_size(
    const Tensor& in,
    int64_t k,
    int64_t dim,
    Tensor::SizesType* target_size,
    size_t* target_dim) {
  *target_dim = in.dim();
  for (size_t i = 0; i < *target_dim; ++i) {
    if (i == dim) {
      target_size[i] = k;
    } else {
      target_size[i] = in.size(i);
    }
  }
  return true;
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

  bool use_partial_sort = k * 64 <= dim_size;

  // Loop through all outer dimensions
  for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    size_t outer_in = outer_idx * outer_stride_in;
    size_t outer_out = outer_idx * outer_stride_out;
    // Loop through all inner dimensions
    for (size_t inner_idx = 0; inner_idx < dim_stride; ++inner_idx) {
      size_t base_in = outer_in + inner_idx;
      size_t base_out = outer_out + inner_idx;

      // Populate the queue with the values from the input tensor
      for (size_t i = 0; i < dim_size; ++i) {
        size_t in_ix = base_in + i * dim_stride;
        queue[i].first = in_data[in_ix];
        queue[i].second = i;
      }

      // Perform topk on the queue
      if (use_partial_sort) {
        if (largest) {
          std::partial_sort(
              queue,
              queue + k,
              queue + dim_size,
              [](const elem_t& x, const elem_t& y) -> bool {
                return (
                    (std::isnan(x.first) && !std::isnan(y.first)) ||
                    (x.first > y.first));
              });
        } else {
          std::partial_sort(
              queue,
              queue + k,
              queue + dim_size,
              [](const elem_t& x, const elem_t& y) -> bool {
                return (
                    (!std::isnan(x.first) && std::isnan(y.first)) ||
                    (x.first < y.first));
              });
        }
      } else {
        if (largest) {
          std::nth_element(
              queue,
              queue + k - 1,
              queue + dim_size,
              [](const elem_t& x, const elem_t& y) -> bool {
                return (
                    (std::isnan(x.first) && !std::isnan(y.first)) ||
                    (x.first > y.first));
              });
          if (sorted) {
            std::sort(
                queue,
                queue + k - 1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return (
                      (std::isnan(x.first) && !std::isnan(y.first)) ||
                      (x.first > y.first));
                });
          }
        } else {
          std::nth_element(
              queue,
              queue + k - 1,
              queue + dim_size,
              [](const elem_t& x, const elem_t& y) -> bool {
                return (
                    (!std::isnan(x.first) && std::isnan(y.first)) ||
                    (x.first < y.first));
              });
          if (sorted) {
            std::sort(
                queue,
                queue + k - 1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return (
                      (!std::isnan(x.first) && std::isnan(y.first)) ||
                      (x.first < y.first));
                });
          }
        }
      }

      // Write the topk values and indices to the output tensors
      for (size_t i = 0; i < k; ++i) {
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

  ET_SWITCH_REALH_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
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
