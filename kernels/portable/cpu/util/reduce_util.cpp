/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>
#include <cstring>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

//
// Helper Functions
//

// Normalize the dimension by adding in_dim if d < 0; for 0-D, clamp to 0
inline size_t _normalize_non_neg_d(ssize_t d, ssize_t in_dim) {
  if (in_dim == 0 && (d == 0 || d == -1)) {
    return 0;
  }
  if (d < 0) {
    return d + in_dim;
  }
  return d;
}

ET_NODISCARD bool check_dim_list_is_valid(
    const exec_aten::Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list) {
  if (dim_list.has_value() && dim_list.value().size() != 0) {
    const auto& reduce_dims = dim_list.value();
    bool dim_exist[kTensorDimensionLimit];
    memset(dim_exist, false, sizeof(dim_exist));
    for (const auto& d : reduce_dims) {
      if (in.dim() == 0) {
        ET_LOG_AND_RETURN_IF_FALSE(d == 0 || d == -1);
      } else {
        ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(d, in.dim()));
      }

      const size_t non_neg_d = _normalize_non_neg_d(d, in.dim());
      ET_LOG_AND_RETURN_IF_FALSE(
          non_neg_d < kTensorDimensionLimit && non_neg_d >= 0);

      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          dim_exist[non_neg_d] == false,
          "dim %zd appears multiple times in the list of dims",
          non_neg_d);
      dim_exist[non_neg_d] = true;
    }
  }

  return true;
}

bool check_dim_in_dim_list(
    const size_t dim,
    const size_t max_dim,
    const exec_aten::ArrayRef<int64_t>& dim_list) {
  for (const auto& d : dim_list) {
    const size_t non_neg_dim = _normalize_non_neg_d(d, max_dim);
    if (dim == non_neg_dim) {
      return true;
    }
  }
  return false;
}

/**
 * Returns the product of the sizes of all reduction dims.
 */
size_t get_reduced_dim_product(
    const Tensor& in,
    const exec_aten::optional<int64_t>& dim) {
  if (in.dim() == 0) {
    return 1;
  }
  size_t dim_product = 1;
  if (!dim.has_value()) {
    for (size_t i = 0; i < in.dim(); ++i) {
      dim_product *= in.size(i);
    }
    return dim_product;
  }
  const size_t d = _normalize_non_neg_d(dim.value(), in.dim());
  return in.size(d);
}

/**
 * Returns the product of the sizes of all reduction dims.
 */
size_t get_reduced_dim_product(
    const Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list) {
  if (in.dim() == 0) {
    return 1;
  }
  size_t dim_product = 1;
  const size_t in_dim = in.dim();
  if (!dim_list.has_value() || dim_list.value().size() == 0) {
    for (size_t i = 0; i < in.dim(); ++i) {
      dim_product *= in.size(i);
    }
    return dim_product;
  }
  for (const auto& d : dim_list.value()) {
    const size_t non_neg_d = _normalize_non_neg_d(d, in_dim);
    dim_product *= in.size(non_neg_d);
  }
  return dim_product;
}

/**
 * Returns the number of elements of the output of reducing `in`
 * over `dim`.
 */
size_t get_out_numel(
    const Tensor& in,
    const exec_aten::optional<int64_t>& dim) {
  size_t out_numel = 1;
  if (dim.has_value()) {
    const auto dim_val = dim.value();
    if (in.dim() == 0) {
      ET_CHECK(dim_val == 0 || dim_val == -1);
    } else {
      ET_CHECK_VALID_DIM(dim_val, in.dim());
    }
    const size_t non_neg_dim = _normalize_non_neg_d(dim_val, in.dim());
    for (size_t d = 0; d < in.dim(); ++d) {
      if (d != non_neg_dim) {
        out_numel *= in.size(d);
      }
    }
  }
  return out_numel;
}

/**
 * Returns the number of elements of the output of reducing `in`
 * over `dim_list`.
 */
size_t get_out_numel(
    const Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list) {
  size_t out_numel = 1;
  if (dim_list.has_value() && dim_list.value().size() != 0) {
    for (size_t d = 0; d < in.dim(); ++d) {
      if (!check_dim_in_dim_list(d, in.dim(), dim_list.value())) {
        out_numel *= in.size(d);
      }
    }
  }
  return out_numel;
}

/**
 * Returns the index of the first element in `in` that maps to `out_ix` when
 * reducing over `dim`. If `dim` is empty, returns `0`.
 */
size_t get_init_index(
    const Tensor& in,
    const exec_aten::optional<int64_t>& dim,
    const size_t out_ix) {
  if (!dim.has_value()) {
    return 0;
  }
  const auto dim_val = dim.value();
  if (in.dim() == 0) {
    ET_CHECK(dim_val == 0 || dim_val == -1);
  } else {
    ET_CHECK_VALID_DIM(dim_val, in.dim());
  }
  const size_t non_neg_dim = _normalize_non_neg_d(dim_val, in.dim());
  size_t init_ix = 0;
  size_t mutable_out_ix = out_ix;
  auto strides = in.strides();
  for (int64_t d = in.dim() - 1; d >= 0; d--) {
    if (d != non_neg_dim) {
      init_ix += (mutable_out_ix % in.size(d)) * strides[d];
      mutable_out_ix /= in.size(d);
    }
  }
  return init_ix;
}

/**
 * Returns the index of the first element in `in` that maps to `out_ix` when
 * reducing over the list of dimensions in `dim_list`. If `dim_list` is null
 * or empty, returns `0`
 */
size_t get_init_index(
    const Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list,
    const size_t out_ix) {
  if (!dim_list.has_value() || dim_list.value().size() == 0) {
    return 0;
  }
  size_t init_ix = 0;
  size_t mutable_out_ix = out_ix;
  auto strides = in.strides();
  for (int64_t d = in.dim() - 1; d >= 0; d--) {
    if (!check_dim_in_dim_list(d, in.dim(), dim_list.value())) {
      init_ix += (mutable_out_ix % in.size(d)) * strides[d];
      mutable_out_ix /= in.size(d);
    }
  }
  return init_ix;
}

//
// Resize out tensor of reduction op
//

size_t compute_reduced_out_size(
    const Tensor& in,
    const exec_aten::optional<int64_t>& dim,
    bool keepdim,
    exec_aten::SizesType* sizes_arr) {
  const auto in_dim = in.dim();
  size_t out_dim = in_dim;

  if (dim.has_value()) {
    const auto dim_val = dim.value();
    const size_t non_neg_dim = _normalize_non_neg_d(dim_val, in_dim);
    for (ssize_t i = 0; i < non_neg_dim; ++i) {
      sizes_arr[i] = in.size(i);
    }
    if (keepdim) {
      sizes_arr[non_neg_dim] = 1;
      for (ssize_t i = non_neg_dim + 1; i < in_dim; ++i) {
        sizes_arr[i] = in.size(i);
      }
    } else {
      for (ssize_t i = non_neg_dim; i < in_dim - 1; ++i) {
        sizes_arr[i] = in.size(i + 1);
      }
      out_dim = in_dim == 0 ? 0 : in_dim - 1;
    }
  } else {
    if (keepdim) {
      for (size_t i = 0; i < in_dim; ++i) {
        sizes_arr[i] = 1;
      }
    } else {
      out_dim = 0;
    }
  }
  return out_dim;
}

size_t compute_reduced_out_size(
    const Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    exec_aten::SizesType* sizes_arr) {
  const auto in_dim = in.dim();
  size_t out_dim = in_dim;

  if (dim_list.has_value() && dim_list.value().size() != 0) {
    const auto& reduce_dims = dim_list.value();
    if (keepdim) {
      for (size_t i = 0; i < in_dim; ++i) {
        if (check_dim_in_dim_list(i, in_dim, reduce_dims)) {
          sizes_arr[i] = 1;
        } else {
          sizes_arr[i] = in.size(i);
        }
      }
    } else {
      size_t out_i = 0;
      for (size_t in_i = 0; in_i < in_dim; ++in_i) {
        if (!check_dim_in_dim_list(in_i, in_dim, reduce_dims)) {
          sizes_arr[out_i] = in.size(in_i);
          out_i++;
        }
      }
      out_dim = out_i;
    }
  } else {
    if (keepdim) {
      for (size_t i = 0; i < in_dim; ++i) {
        sizes_arr[i] = 1;
      }
    } else {
      out_dim = 0;
    }
  }
  return out_dim;
}

Error resize_reduction_out(
    const Tensor& in,
    const exec_aten::optional<int64_t>& dim,
    bool keepdim,
    Tensor& out) {
  exec_aten::SizesType sizes_arr[kTensorDimensionLimit];
  const auto out_dim = compute_reduced_out_size(in, dim, keepdim, sizes_arr);
  exec_aten::ArrayRef<exec_aten::SizesType> out_size{
      sizes_arr, static_cast<size_t>(out_dim)};
  return resize_tensor(out, out_size);
}

Error resize_reduction_out(
    const Tensor& in,
    const exec_aten::optional<exec_aten::ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    Tensor& out) {
  exec_aten::SizesType sizes_arr[kTensorDimensionLimit];
  const auto out_dim =
      compute_reduced_out_size(in, dim_list, keepdim, sizes_arr);
  exec_aten::ArrayRef<exec_aten::SizesType> out_size{
      sizes_arr, static_cast<size_t>(out_dim)};
  return resize_tensor(out, out_size);
}

#ifndef USE_ATEN_LIB

/**
 * Check the validity of arguments for reduction operators.
 */
bool check_reduction_args(
    const Tensor& in,
    const optional<ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  if (dtype.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(dtype.value() == out.scalar_type());
  }
  ET_LOG_AND_RETURN_IF_FALSE(check_dim_list_is_valid(in, dim_list));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));

  return true;
}

/**
 * Check the validity of arguments for reduction operators that take
 * a single dimension argument.
 */
bool check_reduction_args_single_dim(
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out,
    bool allow_empty_dim) {
  if (dtype.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(dtype.value() == out.scalar_type());
  }
  if (in.dim() == 0) {
    if (dim.has_value()) {
      ET_LOG_AND_RETURN_IF_FALSE(dim.value() == 0 || dim.value() == -1);
    }
    return true;
  }

  if (dim.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(dim.value(), in.dim()));
    if (!allow_empty_dim) {
      ET_LOG_AND_RETURN_IF_FALSE(tensor_has_non_empty_dim(in, dim.value()));
    }
  }

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));

  return true;
}

bool check_mean_dim_args(
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(
      check_reduction_args(in, dim_list, keepdim, dtype, out));

  if (dtype) {
    ET_LOG_AND_RETURN_IF_FALSE(torch::executor::isFloatingType(dtype.value()));
    ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == dtype.value());
  } else {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(in));
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(out));
  }

  return true;
}

bool check_amin_amax_args(
    const Tensor& in,
    ArrayRef<int64_t> dim_list,
    bool keepdim,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(
      check_reduction_args(in, dim_list, keepdim, {}, out));
  ET_LOG_AND_RETURN_IF_FALSE(in.scalar_type() == out.scalar_type());

  return true;
}

bool check_argmin_argmax_args(
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(
      check_reduction_args_single_dim(in, dim, keepdim, {}, out));

  ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == ScalarType::Long);

  return true;
}

bool check_min_max_args(
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_indices) {
  ET_LOG_AND_RETURN_IF_FALSE(
      check_reduction_args_single_dim(in, dim, keepdim, {}, max));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, max));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_shape(max, max_indices));
  ET_LOG_AND_RETURN_IF_FALSE(
      tensor_is_default_or_channels_last_dim_order(max_indices));
  ET_LOG_AND_RETURN_IF_FALSE(max_indices.scalar_type() == ScalarType::Long);

  return true;
}

bool check_prod_out_args(
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out) {
  if (dtype.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(dtype.value() == out.scalar_type());
  } else if (isIntegralType(in.scalar_type(), /*includeBool*/ true)) {
    ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == ScalarType::Long);
  } else {
    ET_LOG_AND_RETURN_IF_FALSE(out.scalar_type() == in.scalar_type());
  }

  return true;
}

#endif

} // namespace executor
} // namespace torch
