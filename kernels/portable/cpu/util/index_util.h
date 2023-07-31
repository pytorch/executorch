/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

namespace {

/**
 * For integral index tensors, check that all elements of the index tensor is
 * within the bounds of the tensor being indexed.
 */
template <typename CTYPE>
void check_index_values(
    const Tensor& tensor,
    const size_t dim,
    const Tensor& index) {
  const CTYPE* const index_ptr = index.const_data_ptr<CTYPE>();
  for (size_t i = 0; i < index.numel(); i++) {
    CTYPE index_val = index_ptr[i];
    if (index_val < 0) {
      index_val += tensor.size(dim);
    }
    ET_CHECK_MSG(
        index_val >= 0 && index_val < tensor.size(dim),
        "index[i] %" PRId64 " out of bound [0, input.size(j) %zd)",
        static_cast<int64_t>(index_val),
        tensor.size(i));
  }
}

} // namespace

/**
 * Gets the length of the largest dim of a tensor.
 */
size_t get_max_dim_len(const Tensor& tensor);

/**
 * Checks that the size of a tensor matches an expected size array.
 */
void check_tensor_size(const Tensor& tensor, Tensor::SizesType* expected_size);

/**
 * Counts the number of elements set to true in a boolen index tensor
 */
size_t count_boolean_index(const Tensor& index);

/**
 * Check if indices is represented as a single boolean mask.
 */
bool is_index_mask(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

/**
 * Determines the broadcasting length of indexing tensors in an indices array.
 *
 * For index tensors of integral type, the broadcast length is equal to the
 * number of elements in the tensor.
 *
 * For index tensors of boolean type, the broadcast length is equal to the
 * number of elements set to true.
 *
 * The broadcast length of the indices array is the maximum broadcast length of
 * the index tensors in the array.
 */
size_t get_indices_broadcast_len(
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

/**
 * Indices can be represented as a single boolean mask. The index mask must be
 * the same shape as the tensor being indexed.
 */
void check_indices_mask(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

/**
 * Indices can be represented as an array of optional indexing tensors. An index
 * tensor at the i-th position of the array corresponds to the i-th dimension
 * of the tensor that is being indexed. Each index tensor must conform to the
 * following properties:
 *
 * 1. Each index tensor must be a 1-D tensor or have each other dimension be
 * equal to 1.
 * 2. Each index tensor must be of Int, Long or Bool type (previously the Byte
 * type was also supported but it is being deprecated). Index tensors do not
 * have to share a type; for instance an indices array can contain an integral
 * index tensor AND a boolean index tensor.
 * 3. An integral index tensor represents the indices of the corresponding
 * dimension that have been selected. Therefore, it cannot contain any values
 * that exceed the size of the indexed tensor at the corresponding dimension.
 * Negative indexing is allowed.
 * 4. A boolean index tensor must have a number of elements equal to the size of
 * the indexed tensor at the corresponding dimension.
 * 5. The length of each index tensor must be compatible with each other under
 * broadcasting rules. For boolean index tensors, the length is interpreted as
 * the number of elements that are set to true.
 *
 * Consider a 3 dimensional tensor of size {4, 4, 4}. Consider the following
 * indices arrays:
 *
 * 1. {[0, 2],} selects for [0, :, :] and [2, :, :]
 * 2. {[T, F, T, F],} selects for [0, :, :] and [2, :, :]
 * 3. {[0, 2], [1, 3],} selects for [0, 1, :] and [1, 3, :]
 * 4. {[T, F, T, F], [1, 3],} selects for [0, 1, :] and [1, 3, :]
 * 5. {[0, 2], [3],} selects for [0, 3, :] and [0, 3, :]
 */
void check_indices_list(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

void check_indices(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

/**
 * Determines the size of the result tensor given indices
 */
void get_index_result_size(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    Tensor::SizesType* sizes_arr,
    size_t& dim);

/**
 * Validates that the size of a result tensor matches the expected size when
 * applying an indices array to specified tensor.
 */
void check_index_result_size(
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& result);

/**
 * Given indices in the list format, collect the tensor position that is
 * represented at a particular query index.
 *
 * For example, given a 4 dimensional tensor and an indices list of
 * [[1, 3], [2, 4]]
 * query_idx 0 represents the positional region [1, 2, :, :] of the input tensor
 * query_idx 1 represents the positional region [3, 4, :, :] of the input tensor
 *
 * Once the positional region is determined, the offset needed to align memory
 * pointers to the memory region is computed and returned.
 */
size_t get_index_query_pos_offset(
    size_t query_idx,
    const Tensor& tensor,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices);

} // namespace executor
} // namespace torch
