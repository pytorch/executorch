// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;

namespace {

/**
 * Verifies preconditions of permute_copy_out
 */
void check_preconditions(const Tensor& a, IntArrayRef dims, Tensor& out) {
  // Check amount of dims match, see all the numbers in one error message
  ET_CHECK_MSG(
      a.dim() == dims.size() && a.numel() == out.numel() &&
          a.dim() == out.dim(),
      "Dimension a.dim() %zd != dims.size() %zd, or a.numel() %zd != out.numel() %zd or a.dim() %zd != out.dim() %zd.",
      a.dim(),
      dims.size(),
      a.numel(),
      out.numel(),
      a.dim(),
      out.dim());
  ET_CHECK(a.dim() <= kTensorDimensionLimit);
  // Make sure no dimensions are duplicated and all in the range [-a.dim(),
  // a.dim() - 1].
  auto dim_min = -1 * a.dim();
  // Each bit tracks whether that dimension has been seen in dims.
  size_t used_dims = 0;
  for (int i = 0; i < dims.size(); i++) {
    // Convert dimension to a non-negative number. dim_base is in the range
    // [0 .. a.dim() - 1].
    auto dim_base = dims[i] > -1 ? dims[i] : a.dim() + dims[i];
    ET_CHECK_MSG(
        dim_base >= 0 && dim_base < a.dim(),
        "Dimension at location %d out of range (expected to be in range of [%zd, %zd], but got %" PRId64
        ")",
        i,
        dim_min,
        a.dim() - 1,
        dims[i]);
    const size_t mask = 1UL << dim_base;
    ET_CHECK_MSG(
        !(used_dims & mask),
        "Duplicate dim: %" PRId64 " at location %d is not allowed",
        dim_base,
        i);
    used_dims |= mask;
  }
}

/*
 * Increments an N dimensional index like x[0,0,0] to x[0, 0, 1] to x[0, 0, 2]
 * to x[0, 1, 0] to x[0, 1, 1] etc...
 *
 * index: An array of the same size as sizes. This stores the "counter" being
 *        incremented.
 *
 * sizes: The output tensor dimensions. Allows us to compute the offset into
 *        the input tensor.
 *
 * indices: A list of indices into index that contain non-1 dimension values.
 *          This allows us to eliminate an O(dim) factor from the runtime
 *          in case many dimensions have a value of 1.
 *
 * strides: Permuted strides.
 *
 * offset: The computed offset to index into the input tensor's memory array.
 */
void increment_index_and_offset(
    size_t* index,
    const ArrayRef<SizesType> sizes,
    const ArrayRef<size_t> indices,
    size_t* strides,
    size_t& offset) {
  for (size_t j = indices.size(); j > 0; --j) {
    const size_t i = indices[j - 1];

    index[i]++;
    // Impossible to happen at i = 0 due to precondition check before this
    // function is called
    offset += strides[i];
    if (index[i] == sizes[i]) {
      offset -= sizes[i] * strides[i];
      index[i] = 0;
    } else {
      return;
    }
  }
}

/**
 * Permutes the dimensions of 'a' according to the order in 'dims' overwriting
 * `out`.
 *
 * Assumes that the tensors are contiguous, are the same numel, and have the
 * same dtype. CTYPE should be the C type (like `float` or `int`) that matches
 * the dtype of the tensors.
 */
template <class CTYPE>
void permute_tensors(const Tensor& a, IntArrayRef dims, Tensor& out) {
  auto data_a = a.data_ptr<CTYPE>();
  auto data_out = out.data_ptr<CTYPE>();

  size_t a_index[kTensorDimensionLimit];
  memset(a_index, 0, sizeof(a_index));

  size_t new_strides[kTensorDimensionLimit];
  memset(new_strides, 0, sizeof(new_strides));

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

  // Calculate new strides to save some in loop executions.
  for (ssize_t orig_dim = 0; orig_dim < a.dim(); orig_dim++) {
    auto dim_of_stride =
        dims[orig_dim] >= 0 ? dims[orig_dim] : dims[orig_dim] + a.dim();
    new_strides[orig_dim] = a.strides()[dim_of_stride];

    if (out.sizes()[orig_dim] != 1) {
      non_1_dim_indices[num_non_1_dim_indices++] = orig_dim;
    }
  }

  ArrayRef<size_t> indices(non_1_dim_indices, num_non_1_dim_indices);

  // Loop over out and compute location of out_data[out_idx] in a_data.
  size_t a_offset = 0;
  for (ssize_t out_idx = 0; out_idx < out.numel(); out_idx++) {
    data_out[out_idx] = data_a[a_offset];
    increment_index_and_offset(
        a_index, out.sizes(), indices, new_strides, a_offset);
  }
}

void resize_out_tensor(const Tensor& a, IntArrayRef& dims, Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < a.dim(); i++) {
    expected_output_size[i] = a.size(dims.at(i));
  }
  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(a.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in permute_copy_out");
}

} // namespace

/**
 * Permutes the dimensions of 'a' according to the order in 'dims', and copying
 * that mutation into `out` in order to maintain contiguousness.
 *
 * Asserts that dims uniquely identifies every dimension in a, and that out is
 * the same numel as a
 *
 * permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& permute_copy_out(
    RuntimeContext& context,
    const Tensor& a,
    IntArrayRef dims,
    Tensor& out) {
  (void)context;
  resize_out_tensor(a, dims, out);
  check_preconditions(a, dims, out);
#define PERMUTE_TENSORS(ctype, dtype)     \
  case ScalarType::dtype:                 \
    permute_tensors<ctype>(a, dims, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_SCALAR_TYPES(PERMUTE_TENSORS)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", a.scalar_type());
  }

#undef PERMUTE_TENSORS

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
