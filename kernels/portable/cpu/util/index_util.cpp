// Copyright (c) Meta Platforms, Inc. and affiliates.

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

void check_tensor_size(const Tensor& tensor, Tensor::SizesType* expected_size) {
  for (size_t d = 0; d < tensor.dim(); d++) {
    ET_CHECK_MSG(
        tensor.size(d) == expected_size[d],
        "values.size(%zu) %zd != expected_size[%zu] %zd",
        d,
        tensor.size(d),
        d,
        ssize_t(expected_size[d]));
  }
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
    if (mask.scalar_type() == ScalarType::Bool && mask.dim() == tensor.dim()) {
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
      if (index.scalar_type() == ScalarType::Bool) {
        len = count_boolean_index(index);
      } else {
        len = index.numel();
      }
      broadcast_len = broadcast_len < len ? len : broadcast_len;
    }
  }
  return broadcast_len;
}

void check_indices_mask(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  if (indices.size() == 1 && indices[0].has_value()) {
    const Tensor& mask = indices[0].value();
    ET_CHECK_SAME_SHAPE2(tensor, mask);
  }
}

void check_indices_list(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  size_t broadcast_len = get_indices_broadcast_len(indices);

  for (size_t i = 0; i < indices.size(); i++) {
    const Tensor& index = indices[i].value();
    ScalarType idx_type = index.scalar_type();

    ET_CHECK_MSG(
        get_max_dim_len(index) == index.numel(),
        "Each index tensor must have all dims equal to 1 except one");

    // TODO(ssjia): properly support tensor broadcasting
    if (idx_type == ScalarType::Int || idx_type == ScalarType::Long) {
      ET_CHECK_MSG(
          index.numel() == broadcast_len || index.numel() == 1,
          "indices[%zd].numel() %zd cannot broadcast with length %zd",
          i,
          index.numel(),
          broadcast_len);

      if (idx_type == ScalarType::Int) {
        check_index_values<int32_t>(tensor, i, index);
      } else {
        check_index_values<int64_t>(tensor, i, index);
      }
    } else if (idx_type == ScalarType::Bool) {
      ET_CHECK_MSG(
          index.numel() == tensor.size(i),
          "indices[%zd].numel() %zd incompatible with input.size(%zd) %zd",
          i,
          index.numel(),
          i,
          tensor.size(i));

      size_t len = count_boolean_index(index);
      ET_CHECK_MSG(
          len == broadcast_len || len == 1,
          "indices[%zd] true count %zd cannot broadcast with length %zd",
          i,
          len,
          broadcast_len);
    } else {
      ET_CHECK_MSG(
          false,
          "%hhd scalar type is not supported for indices",
          index.scalar_type());
    }
  }
}

void check_indices(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices) {
  if (is_index_mask(tensor, indices)) {
    check_indices_mask(tensor, indices);
  } else {
    check_indices_list(tensor, indices);
  }
}

void get_index_result_size(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor::SizesType* sizes_arr,
    size_t& dim) {
  // If indexing using a boolean mask, the result will be one dimensional with
  // length equal to the number of true elements in the mask.
  if (is_index_mask(tensor, indices)) {
    dim = 1;
    const Tensor& mask = indices[0].value();
    size_t true_count = count_boolean_index(mask);
    sizes_arr[0] = true_count;
    return;
  }

  // If indexing using a list of index tensors, each index tensor corresponds to
  // one dim of the original tensor. These tensors can be broadcasted, so first
  // retrieve broadcasted size of each tensor.
  size_t broadcast_len = get_indices_broadcast_len(indices);
  // The expected ndim of the result tensor is equal to the ndim of the original
  // tensor offset by the number of dimensions that were indexed.
  dim = tensor.dim() - indices.size() + 1;

  // The leading dim of the result should be equal to number of index queries
  sizes_arr[0] = broadcast_len;

  // The remaining dims should match the size of the unqueried dims of original
  // tensor.
  for (size_t i = 1; i < dim; i++) {
    sizes_arr[i] = tensor.size(i + indices.size() - 1);
  }
}

void check_index_result_size(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& result) {
  size_t expected_ndim = 0;
  Tensor::SizesType expected_size[kTensorDimensionLimit];
  get_index_result_size(tensor, indices, expected_size, expected_ndim);

  ET_CHECK_MSG(
      result.dim() == expected_ndim,
      "result.dim() must be %zu, got dim of %zd",
      expected_ndim,
      result.dim());

  check_tensor_size(result, expected_size);
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
    } else if (idx_type == ScalarType::Bool) {
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
      ET_CHECK_MSG(
          index_val < index.numel(), "invalid index_val %zd", index_val);
      // Update data pointers
      offset += index_val * step_len;
    }
  }
  return offset;
}

} // namespace executor
} // namespace torch
