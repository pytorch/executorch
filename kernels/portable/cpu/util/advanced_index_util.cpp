/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

using Tensor = executorch::aten::Tensor;
using TensorOptList = executorch::aten::ArrayRef<std::optional<Tensor>>;

namespace {

bool check_indices_dtypes(TensorOptList indices) {
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      ScalarType ix_type = index.scalar_type();
      ET_CHECK_OR_RETURN_FALSE(
          ix_type == ScalarType::Long || ix_type == ScalarType::Int ||
              ix_type == ScalarType::Byte || ix_type == ScalarType::Bool,
          "Index tensors should be Long, Int, Byte or Bool; got %d",
          static_cast<int>(ix_type));
    }
  }
  return true;
}

bool is_mask_index(const Tensor& index) {
  if (index.scalar_type() == ScalarType::Bool ||
      index.scalar_type() == ScalarType::Byte) {
    return true;
  }
  return false;
}

bool check_mask_indices(const Tensor& in, TensorOptList indices) {
  size_t in_i = 0;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      if (is_mask_index(index)) {
        ET_CHECK_OR_RETURN_FALSE(
            index.dim() > 0, "Zero-dimensional mask index not allowed");
        for (const auto j : c10::irange(index.dim())) {
          if (index.size(j) != in.size(in_i + j)) {
#if ET_LOG_ENABLED
            auto mask_shape = executorch::runtime::tensor_shape_to_c_string(
                executorch::runtime::Span<const Tensor::SizesType>(
                    index.sizes().data(), index.sizes().size()));
            auto input_shape = executorch::runtime::tensor_shape_to_c_string(
                executorch::runtime::Span<const Tensor::SizesType>(
                    in.sizes().data() + in_i, index.sizes().size()));
            ET_LOG(
                Error,
                "The shape of mask index %s must match the sizes of the corresponding input dimensions %s.",
                mask_shape.data(),
                input_shape.data());
#endif // ET_LOG_ENABLED
            return false;
          }
        }
        in_i += index.dim();
      } else {
        in_i += 1;
      }
    } else {
      in_i += 1;
    }
  }
  return true;
}

template <typename CTYPE_IX>
size_t _count_trues_in_mask_index(const Tensor& index) {
  const CTYPE_IX* const index_ptr = index.const_data_ptr<CTYPE_IX>();
  size_t sum = 0;
  for (const auto i : c10::irange(index.numel())) {
    if (index_ptr[i]) {
      sum += 1;
    }
  }
  return sum;
}

size_t count_trues_in_mask_index(const Tensor& index) {
  if (index.scalar_type() == ScalarType::Bool) {
    return _count_trues_in_mask_index<bool>(index);
  } else {
    return _count_trues_in_mask_index<uint8_t>(index);
  }
}

template <typename CTYPE_IX>
void _query_mask_index(const Tensor& index, size_t query_idx, size_t* res) {
  const CTYPE_IX* const index_ptr = index.const_data_ptr<CTYPE_IX>();
  // Broadcasting for mask index tensors
  size_t num_true = _count_trues_in_mask_index<CTYPE_IX>(index);
  if (num_true == 1) {
    query_idx = 0;
  }
  // Extract the index value by finding the idx-th element that is set to
  // true.
  size_t count = 0;
  size_t flat_ix = 0;
  for (const auto i : c10::irange(index.numel())) {
    if (index_ptr[i]) {
      if (count == query_idx) {
        flat_ix = i;
        break;
      } else {
        count++;
      }
    }
  }
  delinearize_index(flat_ix, index, res, kTensorDimensionLimit);
}

void query_mask_index(const Tensor& index, size_t query_idx, size_t* res) {
  if (index.scalar_type() == ScalarType::Bool) {
    _query_mask_index<bool>(index, query_idx, res);
  } else {
    _query_mask_index<uint8_t>(index, query_idx, res);
  }
}

int64_t query_integral_index(
    const Tensor& index,
    size_t* ix_coord,
    size_t broadcast_ndim) {
  size_t flat_ix = linearize_access_indexes(
      {ix_coord, broadcast_ndim}, broadcast_ndim, index);

  ScalarType idx_type = index.scalar_type();
  int64_t index_val = 0;
  // Extract the index value
  if (idx_type == ScalarType::Int) {
    const int32_t* const index_ptr = index.const_data_ptr<int32_t>();
    index_val = static_cast<int64_t>(index_ptr[flat_ix]);
  } else {
    const int64_t* const index_ptr = index.const_data_ptr<int64_t>();
    index_val = index_ptr[flat_ix];
  }
  return index_val;
}

} // namespace

bool check_index_args(const Tensor& in, TensorOptList indices, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(check_indices_dtypes(indices));
  ET_CHECK_OR_RETURN_FALSE(
      static_cast<ssize_t>(indices.size()) <= in.dim(),
      "Indexing too many dimensions");
  ET_LOG_AND_RETURN_IF_FALSE(check_mask_indices(in, indices));
  return true;
}

size_t count_index_blocks(TensorOptList indices) {
  size_t block_count = 0;
  bool in_block = false;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      if (!in_block) {
        in_block = true;
        block_count++;
      }
    } else {
      in_block = false;
    }
  }
  return block_count;
}

bool get_indices_broadcast_shape(
    TensorOptList indices,
    Tensor::SizesType* ix_sizes,
    size_t* ix_ndim) {
  // Holds the (reversed) broadcasted shape of the indices.
  Tensor::SizesType rev_ix_sizes[kTensorDimensionLimit];
  ssize_t curr_ndim = 0;

  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      if (is_mask_index(index)) {
        Tensor::SizesType len =
            static_cast<Tensor::SizesType>(count_trues_in_mask_index(index));
        if (curr_ndim == 0) {
          curr_ndim = 1;
          rev_ix_sizes[0] = len;
        } else if (rev_ix_sizes[0] == 1) {
          rev_ix_sizes[0] = len;
        } else if (len != 1 && rev_ix_sizes[0] != len) {
          ET_CHECK_OR_RETURN_FALSE(false, "Broadcast of mask index failed.");
        }
      } else {
        for (const auto j : c10::irange(index.dim())) {
          Tensor::SizesType rev_j_size =
              static_cast<Tensor::SizesType>(index.size(index.dim() - j - 1));
          if (j >= curr_ndim) {
            curr_ndim = j + 1;
            rev_ix_sizes[j] = rev_j_size;
          } else if (rev_ix_sizes[j] == 1) {
            rev_ix_sizes[j] = rev_j_size;
          } else if (rev_j_size != 1 && rev_ix_sizes[j] != rev_j_size) {
            ET_CHECK_OR_RETURN_FALSE(false, "Broadcast of index failed.");
          }
        }
      }
    }
  }

  for (const auto i : c10::irange(curr_ndim)) {
    ix_sizes[i] = rev_ix_sizes[curr_ndim - i - 1];
  }
  (*ix_ndim) = curr_ndim;
  return true;
}

size_t get_indices_broadcast_ndim(TensorOptList indices) {
  ssize_t ndim = 0;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      if (is_mask_index(index)) {
        if (ndim == 0) {
          ndim = 1;
        }
      } else {
        if (ndim < index.dim()) {
          ndim = index.dim();
        }
      }
    }
  }
  return ndim;
}

size_t get_num_indexed_dims(TensorOptList indices) {
  size_t num_indexed_dims = 0;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      if (is_mask_index(index)) {
        num_indexed_dims += index.dim();
      } else {
        num_indexed_dims += 1;
      }
    }
  }
  return num_indexed_dims;
}

size_t get_num_null_indices(TensorOptList indices) {
  size_t num_null_indices = 0;
  for (const auto i : c10::irange(indices.size())) {
    if (!indices[i].has_value()) {
      num_null_indices += 1;
    }
  }
  return num_null_indices;
}

size_t get_num_leading_null_indices(TensorOptList indices) {
  size_t start = 0;
  while (!indices[start].has_value()) {
    start += 1;
  }
  return start;
}

bool get_index_out_target_size(
    const Tensor& in,
    TensorOptList indices,
    bool adjacent,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  Tensor::SizesType broadcast_sizes[kTensorDimensionLimit];
  size_t broadcast_ndim = 0;
  if (!get_indices_broadcast_shape(indices, broadcast_sizes, &broadcast_ndim)) {
    return false;
  }

  size_t num_null_indices = get_num_null_indices(indices);
  size_t num_indexed_dims = get_num_indexed_dims(indices);

  ET_CHECK_OR_RETURN_FALSE(
      static_cast<ssize_t>(num_null_indices + num_indexed_dims) <= in.dim(),
      "Indexing too many dimensions; num_null_indices = %zu, num_indexed_dims = %zu, in.dim() = %" ET_PRI_TENSOR_DIM,
      num_null_indices,
      num_indexed_dims,
      in.dim());

  ET_CHECK_OR_RETURN_FALSE(
      in.dim() + broadcast_ndim - num_indexed_dims <= kTensorDimensionLimit,
      "Out tensor would exceed number of allowed dimensions; in.dim() = %" ET_PRI_TENSOR_DIM
      ", broadcast_ndim = %zu, num_indexed_dims = %zu, kTensorDimensionLimit = %zu",
      in.dim(),
      broadcast_ndim,
      num_indexed_dims,
      kTensorDimensionLimit);

  (*out_ndim) = in.dim() + broadcast_ndim - num_indexed_dims;

  if (adjacent) {
    size_t start = get_num_leading_null_indices(indices);
    for (const auto i : c10::irange(start)) {
      out_sizes[i] = in.size(i);
    }
    for (const auto i : c10::irange(broadcast_ndim)) {
      out_sizes[i + start] = broadcast_sizes[i];
    }
    for (const auto i : c10::irange(num_indexed_dims + start, in.dim())) {
      out_sizes[i + broadcast_ndim - num_indexed_dims] = in.size(i);
    }
  } else {
    for (const auto i : c10::irange(broadcast_ndim)) {
      out_sizes[i] = broadcast_sizes[i];
    }
    size_t in_i = 0;
    size_t out_i = broadcast_ndim;
    for (const auto i : c10::irange(indices.size())) {
      if (!indices[i].has_value()) {
        out_sizes[out_i++] = in.size(in_i++);
      } else {
        const Tensor& index = indices[i].value();
        if (is_mask_index(index)) {
          in_i += index.dim();
        } else {
          in_i += 1;
        }
      }
    }
    for (const auto i :
         c10::irange(num_indexed_dims + num_null_indices, in.dim())) {
      out_sizes[i + broadcast_ndim - num_indexed_dims] = in.size(i);
    }
  }
  return true;
}

// dim_map maps non-indexed input dimensions to the corresponding output
// dimensions. Indexed dimensions are mapped to -1.
void compute_dim_map(
    const Tensor& in,
    TensorOptList indices,
    int32_t* dim_map,
    bool adjacent) {
  size_t broadcast_ndim = get_indices_broadcast_ndim(indices);
  size_t start = get_num_leading_null_indices(indices);
  size_t num_indexed_dims = get_num_indexed_dims(indices);
  size_t num_null_indices = get_num_null_indices(indices);

  if (adjacent) {
    for (const auto i : c10::irange(start)) {
      dim_map[i] = i;
    }
    for (const auto i : c10::irange(start, start + num_indexed_dims)) {
      dim_map[i] = -1;
    }
    for (const auto i : c10::irange(start + num_indexed_dims, in.dim())) {
      dim_map[i] = i - num_indexed_dims + broadcast_ndim;
    }
  } else {
    size_t in_i = 0;
    size_t out_i = broadcast_ndim;
    for (const auto i : c10::irange(indices.size())) {
      if (!indices[i].has_value()) {
        dim_map[in_i++] = out_i++;
      } else {
        const Tensor& index = indices[i].value();
        if (is_mask_index(index)) {
          for ([[maybe_unused]] const auto j : c10::irange(index.dim())) {
            dim_map[in_i++] = -1;
          }
        } else {
          dim_map[in_i++] = -1;
        }
      }
    }
    for (const auto i :
         c10::irange(num_indexed_dims + num_null_indices, in.dim())) {
      dim_map[i] = i - num_indexed_dims + broadcast_ndim;
    }
  }
}

// ix_map maps indexed input dimensions to the corresponding index.
// Non-indexed dimensions are mapped to -1.
void compute_index_map(
    const Tensor& in,
    TensorOptList indices,
    int32_t* ix_map) {
  for (const auto i : c10::irange(in.dim())) {
    ix_map[i] = -1;
  }
  size_t in_i = 0;
  for (const auto i : c10::irange(indices.size())) {
    if (indices[i].has_value()) {
      const Tensor& index = indices[i].value();
      if (is_mask_index(index)) {
        for ([[maybe_unused]] const auto j : c10::irange(index.dim())) {
          ix_map[in_i++] = i;
        }
      } else {
        ix_map[in_i++] = i;
      }
    } else {
      in_i++;
    }
  }
}

bool get_in_coord(
    const Tensor& in,
    TensorOptList indices,
    size_t start,
    size_t broadcast_ndim,
    int32_t* dim_map,
    int32_t* ix_map,
    size_t* out_coord,
    size_t* in_coord) {
  for (ssize_t i = 0; i < in.dim(); i++) {
    if (dim_map[i] >= 0) {
      in_coord[i] = out_coord[dim_map[i]];
    } else {
      const Tensor& index = indices[ix_map[i]].value();

      size_t ix_coord[kTensorDimensionLimit];
      for (const auto j : c10::irange(broadcast_ndim)) {
        ix_coord[j] = out_coord[j + start];
      }

      if (is_mask_index(index)) {
        size_t query_ix = ix_coord[broadcast_ndim - 1];
        size_t query_result[kTensorDimensionLimit];
        query_mask_index(index, query_ix, query_result);
        for (const auto j : c10::irange(index.dim())) {
          in_coord[i + j] = query_result[j];
        }
        i += index.dim() - 1;
      } else {
        int64_t index_val =
            query_integral_index(index, ix_coord, broadcast_ndim);
        if (index_val < 0) {
          index_val += in.size(i);
        }
        ET_CHECK_OR_RETURN_FALSE(
            index_val >= 0 && index_val < in.size(i),
            "Index %" PRId64
            " is out of bounds for input dimension %zd with size %zd.",
            index_val,
            i,
            in.size(i));
        in_coord[i] = static_cast<size_t>(index_val);
      }
    }
  }
  return true;
}

std::pair<size_t, bool> get_in_ix(
    const Tensor& in,
    TensorOptList indices,
    Tensor& out,
    size_t out_ix,
    size_t start,
    size_t broadcast_ndim,
    int32_t* dim_map,
    int32_t* ix_map) {
  size_t out_coord[kTensorDimensionLimit];
  delinearize_index(out_ix, out, out_coord, kTensorDimensionLimit);

  size_t in_coord[kTensorDimensionLimit];
  bool success = get_in_coord(
      in, indices, start, broadcast_ndim, dim_map, ix_map, out_coord, in_coord);
  if (!success) {
    return std::make_pair(0, false);
  }
  return std::make_pair(coordinateToIndex(in, in_coord), true);
}

} // namespace executor
} // namespace torch
