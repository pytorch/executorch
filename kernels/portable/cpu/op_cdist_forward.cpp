/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/distance_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::optional;
using exec_aten::Tensor;

namespace {

inline ArrayRef<Tensor::SizesType> get_batch_sizes(const Tensor& tensor) {
  return {tensor.sizes().data(), tensor.sizes().size() - 2};
}

template <typename CTYPE, typename Norm>
void cdist(const Tensor& x1, const Tensor& x2, Tensor& out, double p) {
  if (out.numel() == 0) {
    return;
  }

  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  // If the last dimension of x1 (which is equal to the last dimension of x2)
  // has size 0, then the output is filled with 0s.
  if (x1.numel() == 0) {
    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      out_data[out_ix] = 0;
    }
    return;
  }

  const CTYPE* x1_data = x1.const_data_ptr<CTYPE>();
  const CTYPE* x2_data = x2.const_data_ptr<CTYPE>();

  const ArrayRef<Tensor::SizesType> x1_batch_sizes = get_batch_sizes(x1);
  const ArrayRef<Tensor::SizesType> x2_batch_sizes = get_batch_sizes(x2);
  const ArrayRef<Tensor::SizesType> out_batch_sizes = get_batch_sizes(out);

  const bool x1_is_broadcasted = !out_batch_sizes.equals(x1_batch_sizes);
  const bool x2_is_broadcasted = !out_batch_sizes.equals(x2_batch_sizes);
  const bool any_is_broadcasted = (x1_is_broadcasted || x2_is_broadcasted);

  size_t out_batch_numel = 1;
  for (auto i : out_batch_sizes) {
    out_batch_numel *= i;
  }

  size_t P = static_cast<size_t>(x1.size(x1.dim() - 2)); // NOLINT
  size_t R = static_cast<size_t>(x2.size(x2.dim() - 2)); // NOLINT
  size_t M = static_cast<size_t>(x1.size(x1.dim() - 1)); // NOLINT

  size_t x1_inner_size = P * M;
  size_t x2_inner_size = R * M;
  size_t out_inner_size = P * R;

  for (size_t b = 0; b < out_batch_numel; ++b) {
    size_t x1_base_ix = b * x1_inner_size;
    size_t x2_base_ix = b * x2_inner_size;
    size_t out_base_ix = b * out_inner_size;

    if (any_is_broadcasted) {
      size_t out_base_coord[kTensorDimensionLimit];
      delinearize_index(
          out_base_ix, out, out_base_coord, kTensorDimensionLimit);

      if (x1_is_broadcasted) {
        x1_base_ix = linearize_access_indexes(out_base_coord, out.dim(), x1);
      }
      if (x2_is_broadcasted) {
        x2_base_ix = linearize_access_indexes(out_base_coord, out.dim(), x2);
      }
    }

    size_t out_ix = 0;
    for (size_t i = 0; i < P; ++i) {
      const CTYPE* row_i = x1_data + x1_base_ix + i * M;
      for (size_t j = 0; j < R; ++j) {
        const CTYPE* row_j = x2_data + x2_base_ix + j * M;
        CTYPE agg = 0;
        for (size_t k = 0; k < M; ++k) {
          CTYPE diff = std::abs(row_i[k] - row_j[k]);
          agg = Norm::reduce(agg, Norm::map(diff, p));
        }
        out_data[out_base_ix + out_ix++] = Norm::finish(agg, p);
      }
    }
  }
}

template <typename CTYPE>
void cdist(const Tensor& x1, const Tensor& x2, Tensor& out, double p) {
  if (p == 0.0) {
    cdist<CTYPE, L0<CTYPE>>(x1, x2, out, p);
  } else if (p == 1.0) {
    cdist<CTYPE, L1<CTYPE>>(x1, x2, out, p);
  } else if (p == 2.0) {
    cdist<CTYPE, L2<CTYPE>>(x1, x2, out, p);
  } else if (p == INFINITY) {
    cdist<CTYPE, Linf<CTYPE>>(x1, x2, out, p);
  } else {
    cdist<CTYPE, Lp<CTYPE>>(x1, x2, out, p);
  }
}

} // namespace

Tensor& _cdist_forward_out(
    KernelRuntimeContext& ctx,
    const Tensor& x1,
    const Tensor& x2,
    double p,
    optional<int64_t> compute_mode,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(x1, x2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(x1), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      check_cdist_args(x1, x2, p, compute_mode, out),
      InvalidArgument,
      out);

  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;

  ET_KERNEL_CHECK(
      ctx,
      get_broadcast_target_size(
          {x1.sizes().data(), x1.sizes().size() - 2},
          {x2.sizes().data(), x2.sizes().size() - 2},
          target_sizes,
          kTensorDimensionLimit,
          &target_ndim) == Error::Ok,
      InvalidArgument,
      out);

  target_ndim += 2;
  target_sizes[target_ndim - 2] = x1.size(x1.dim() - 2);
  target_sizes[target_ndim - 1] = x2.size(x2.dim() - 2);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType out_type = out.scalar_type();
  constexpr auto name = "_cdist_forward.out";

  ET_SWITCH_FLOAT_TYPES(
      out_type, ctx, name, CTYPE, [&] { cdist<CTYPE>(x1, x2, out, p); });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
