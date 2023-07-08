#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>
#include <string.h>

namespace torch {
namespace executor {

using SizesType = exec_aten::SizesType;
using StridesType = exec_aten::StridesType;

/**
 * Returns a tensor that is a transposed version of input in out.
 * The given dimensions dim0 and dim1 are swapped.
 *
 * @param[in] a the input tensor.
 * @param[in] dim0 the first dimension to be transposed
 * @param[in] dim1 the second dimension to be transposed.
 *
 */
template <typename T>
void transpose_tensors(
    const Tensor& a,
    int64_t dim0,
    int64_t dim1,
    Tensor& out);

namespace {
/**
 * Increments an N dimensional index like x[0,0,0] to x[0, 0, 1] to x[0, 0, 2]
 * to x[0, 1, 0] to x[0, 1, 1] etc...
 *
 * @param index: An array of the same size as sizes. This stores the "counter"
 * being incremented.
 *
 * @param new_sizes: The output tensor dimensions. Allows us to compute the
 * offset into the input tensor.
 *
 * @param non_one_indices: A list of indices into index that contain non-1
 * dimension values. This allows us to eliminate an O(dim) factor from the
 * runtime in case many dimensions have a value of 1.
 *
 * @param new_strides: Strides corresponding to new_sizes.
 *
 * @param offset: The computed offset to index into the input tensor's memory
 * array.
 */
inline void increment_index_and_offset(
    size_t* index,
    const SizesType* new_sizes,
    const StridesType* new_strides,
    const ArrayRef<size_t> non_one_indices,
    size_t& offset) {
  for (size_t j = non_one_indices.size(); j > 0; --j) {
    const size_t i = non_one_indices[j - 1];

    index[i]++;
    // Impossible to happen at i = 0 due to precondition check before this
    // function is called
    offset += new_strides[i];
    if (index[i] == new_sizes[i]) {
      offset -= new_sizes[i] * new_strides[i];
      index[i] = 0;
    } else {
      return;
    }
  }
}

} // namespace

template <typename T>
void transpose_tensors(
    const Tensor& a,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  auto dim = a.dim();
  auto data_a = a.const_data_ptr<T>();
  auto data_out = out.mutable_data_ptr<T>();
  auto a_sizes = a.sizes();
  auto a_strides = a.strides();

  size_t out_index[kTensorDimensionLimit];
  memset(out_index, 0, sizeof(out_index));

  StridesType new_strides[kTensorDimensionLimit];
  memcpy(new_strides, a_strides.data(), dim * sizeof(StridesType));

  SizesType new_sizes[kTensorDimensionLimit];
  memcpy(new_sizes, a_sizes.data(), dim * sizeof(SizesType));

  std::swap(new_sizes[dim0], new_sizes[dim1]);
  std::swap(new_strides[dim1], new_strides[dim0]);

  // non_1_dim_indices stores the indices of the dimensions that have a value
  // greater than 1. Dimensions can only have a value of 1 or larger.
  //
  // This list is stored in the increasing order of the output (not input)
  // dimension. i.e. lower index of non-1 output dimension first). This
  // allows us to loop over only the non-1 indices (and skip the ones that
  // have a value of 1 since they don't contribute to any meaningful computation
  // in terms of increasing the number of elements to be copied).
  //
  // We loop over these non-1 indices in the reverse order since we want to
  // process the last output dimension first (to be able to walk the input
  // tensor in output tensor order.
  size_t non_1_dim_indices[kTensorDimensionLimit];
  size_t num_non_1_dim_indices = 0;
  for (size_t cur_dim = 0; cur_dim < dim; cur_dim++) {
    if (new_sizes[cur_dim] != 1) {
      non_1_dim_indices[num_non_1_dim_indices++] = cur_dim;
    }
  }

  ArrayRef<size_t> indices(non_1_dim_indices, num_non_1_dim_indices);

  // Loop over and copy input elements into output
  size_t a_offset = 0;
  for (ssize_t out_offset = 0; out_offset < a.numel(); out_offset++) {
    data_out[out_offset] = data_a[a_offset];
    increment_index_and_offset(
        out_index, new_sizes, new_strides, indices, a_offset);
  }
}

} // namespace executor
} // namespace torch
