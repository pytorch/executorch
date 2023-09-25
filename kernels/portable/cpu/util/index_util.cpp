/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/index_util.h>

namespace torch {
namespace executor {

size_t get_max_dim_len(const Tensor& tensor) {
  size_t dim_len = 0;
  for (size_t i = 0; i < tensor.dim(); ++i) {
    dim_len = dim_len < tensor.size(i) ? tensor.size(i) : dim_len;
  }
  return dim_len;
}

size_t count_boolean_index(const Tensor& index) {
  const bool* const index_ptr = index.const_data_ptr<bool>();
  size_t sum = 0;
  for (size_t i = 0; i < index.numel(); ++i) {
    if (index_ptr[i]) {
      sum += 1;
    }
  }
  return sum;
}

bool is_index_mask(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  if (indices.size() == 1 && indices[0].has_value()) {
    const Tensor& mask = indices[0].value();
    if ((mask.scalar_type() == ScalarType::Bool ||
         mask.scalar_type() == ScalarType::Byte) &&
        mask.dim() == tensor.dim()) {
      return true;
    }
  }
  return false;
}

size_t get_indices_broadcast_len(
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  size_t broadcast_len = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      size_t len = 0;
      if (index.scalar_type() == ScalarType::Bool ||
          index.scalar_type() == ScalarType::Byte) {
        len = count_boolean_index(index);
      } else {
        len = index.numel();
      }
      broadcast_len = broadcast_len < len ? len : broadcast_len;
    }
  }
  return broadcast_len;
}

bool indices_mask_is_valid(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  if (indices.size() == 1 && indices[0].has_value()) {
    const Tensor& mask = indices[0].value();
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_shape(tensor, mask));
  }
  return true;
}

bool indices_list_is_valid(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  size_t broadcast_len = get_indices_broadcast_len(indices);

  for (size_t i = 0; i < indices.size(); i++) {
    const Tensor& index = indices[i].value();
    ScalarType idx_type = index.scalar_type();

    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        get_max_dim_len(index) == index.numel(),
        "Each index tensor must have all dims equal to 1 except one");

    // TODO(ssjia): properly support tensor broadcasting
    if (idx_type == ScalarType::Int || idx_type == ScalarType::Long) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          index.numel() == broadcast_len || index.numel() == 1,
          "indices[%zd].numel() %zd cannot broadcast with length %zd",
          i,
          index.numel(),
          broadcast_len);

      if (idx_type == ScalarType::Int) {
        ET_LOG_AND_RETURN_IF_FALSE(
            index_values_are_valid<int32_t>(tensor, i, index));
      } else {
        ET_LOG_AND_RETURN_IF_FALSE(
            index_values_are_valid<int64_t>(tensor, i, index));
      }
    } else if (idx_type == ScalarType::Bool || idx_type == ScalarType::Byte) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          index.numel() == tensor.size(i),
          "indices[%zd].numel() %zd incompatible with input.size(%zd) %zd",
          i,
          index.numel(),
          i,
          tensor.size(i));

      size_t len = count_boolean_index(index);
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          len == broadcast_len || len == 1,
          "indices[%zd] true count %zd cannot broadcast with length %zd",
          i,
          len,
          broadcast_len);
    } else {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          false,
          "%" PRId8 " scalar type is not supported for indices",
          static_cast<int8_t>(index.scalar_type()));
    }
  }
  return true;
}

bool indices_are_valid(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  if (is_index_mask(tensor, indices)) {
    return indices_mask_is_valid(tensor, indices);
  } else {
    return indices_list_is_valid(tensor, indices);
  }
}

size_t get_index_query_pos_offset(
    size_t query_idx,
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  size_t offset = 0;
  for (size_t dim = 0; dim < indices.size(); dim++) {
    const Tensor& index = indices[dim].value();
    ScalarType idx_type = index.scalar_type();

    size_t step_len = getTrailingDims(tensor, static_cast<int64_t>(dim));
    if (idx_type == ScalarType::Int || idx_type == ScalarType::Long) {
      // Apply broadcasting, if needed
      size_t adjusted_idx = query_idx < index.numel() ? query_idx : 0u;
      int64_t index_val = 0;
      // Extract the index value
      if (idx_type == ScalarType::Int) {
        const int32_t* const index_ptr = index.const_data_ptr<int32_t>();
        index_val = static_cast<int64_t>(index_ptr[adjusted_idx]);
      } else {
        const int64_t* const index_ptr = index.const_data_ptr<int64_t>();
        index_val = index_ptr[adjusted_idx];
      }
      // Adjust the index value and update data pointers
      if (index_val < 0) {
        index_val += tensor.size(dim);
      }
      offset += index_val * step_len;
    } else if (idx_type == ScalarType::Bool || idx_type == ScalarType::Byte) {
      const bool* const index_ptr = index.const_data_ptr<bool>();
      // Broadcasting for boolean index tensors
      size_t num_true = count_boolean_index(index);
      size_t adjusted_idx = query_idx < num_true ? query_idx : 0u;
      // Extract the index value by finding the idx-th element that is set to
      // true.
      size_t count = 0;
      size_t index_val = index.numel();
      for (size_t i = 0; i < index.numel(); ++i) {
        if (index_ptr[i]) {
          if (count == adjusted_idx) {
            index_val = i;
            break;
          } else {
            count++;
          }
        }
      }

      // Update data pointers
      offset += index_val * step_len;
    }
  }
  return offset;
}

bool check_index_args(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor& out) {
  // size of indices must not exceed the number of dimensions
  if (indices.size() > 0) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, indices.size() - 1));
  }
  ET_LOG_AND_RETURN_IF_FALSE(indices_are_valid(in, indices));
  return true;
}

void get_index_out_target_size(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  // If indexing using a boolean mask, the result will be one dimensional with
  // length equal to the number of true elements in the mask.
  if (is_index_mask(in, indices)) {
    *out_ndim = 1;
    const Tensor& mask = indices[0].value();
    size_t true_count = count_boolean_index(mask);
    out_sizes[0] = true_count;
    return;
  }

  // If indexing using a list of index tensors, each index tensor corresponds to
  // one dim of the original tensor. These tensors can be broadcasted, so first
  // retrieve broadcasted size of each tensor.
  size_t broadcast_len = get_indices_broadcast_len(indices);
  // The expected ndim of the result tensor is equal to the ndim of the original
  // tensor offset by the number of dimensions that were indexed.
  *out_ndim = in.dim() - indices.size() + 1;

  // The leading dim of the result should be equal to number of index queries
  out_sizes[0] = broadcast_len;

  // The remaining dims should match the size of the unqueried dims of original
  // tensor.
  for (size_t i = 1; i < *out_ndim; i++) {
    out_sizes[i] = in.size(i + indices.size() - 1);
  }
}

bool check_index_put_args(
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    Tensor& out) {
  // size of indices must not exceed the number of dimensions
  if (indices.size() > 0) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, indices.size() - 1));
  }

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, values, out));

  ET_LOG_AND_RETURN_IF_FALSE(indices_are_valid(in, indices));

  size_t expected_ndim = 0;
  Tensor::SizesType expected_size[kTensorDimensionLimit];
  get_index_out_target_size(in, indices, expected_size, &expected_ndim);
  // If values not broadcastable, then check it is equal to the size of the
  // expected indexing result.
  if (values.numel() != 1) {
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_has_expected_size(values, {expected_size, expected_ndim}));
  }

  return true;
}

bool check_index_select_args(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  if (in.dim() == 0) {
    ET_LOG_AND_RETURN_IF_FALSE(dim == 0);
  } else {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  }

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      index.scalar_type() == ScalarType::Long ||
          index.scalar_type() == ScalarType::Int,
      "Expected index to have type of Long or Int, but found %s",
      toString(index.scalar_type()));

  if (index.dim() > 0) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(index, 1));
  }
  if (index.scalar_type() == ScalarType::Long) {
    const int64_t* const index_ptr = index.const_data_ptr<int64_t>();
    for (size_t i = 1; i < index.numel(); ++i) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          index_ptr[i] >= 0 && index_ptr[i] < in.size(dim),
          "index[%zu] = %" PRId64 " is out of bounds for in.size(%" PRId64
          ") = %zd",
          i,
          index_ptr[i],
          dim,
          in.size(dim));
    }
  } else {
    const int32_t* const index_ptr = index.const_data_ptr<int32_t>();
    for (size_t i = 1; i < index.numel(); ++i) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          index_ptr[i] >= 0 && index_ptr[i] < in.size(dim),
          "index[%zu] = %" PRId32 " is out of bounds for in.size(%" PRId64
          ") = %zd",
          i,
          index_ptr[i],
          dim,
          static_cast<size_t>(in.size(dim)));
    }
  }

  return true;
}

void get_index_select_out_target_size(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();
  for (size_t i = 0; i < in.dim(); ++i) {
    if (i == dim) {
      out_sizes[i] = index.numel();
    } else {
      out_sizes[i] = in.size(i);
    }
  }
}

} // namespace executor
} // namespace torch
