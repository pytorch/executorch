// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>

#include <executorch/kernels/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

namespace {
void check_args(
    const Tensor& input,
    int64_t split_size,
    int64_t dim,
    TensorList out) {
  ET_CHECK_MSG(
      input.dim() > 0,
      "input must have at least one dimension; saw %zd",
      input.dim());
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  const ssize_t dim_size = input.size(dim);
  ET_CHECK_MSG(
      split_size >= 0,
      "split_size %" PRId64 " must be non-negative",
      split_size);
  ET_CHECK_MSG(
      split_size > 0 || dim_size == 0,
      "split_size is zero but input.size(%" PRId64 ") %zd is non-zero",
      dim,
      dim_size);

  // Check the number of outputs.
  //
  // The specified dimension will be split into split_size-sized chunks, with
  // the final chunk possibly being smaller. So, the expected output length is
  // ceil(dim_size / split_size).
  //
  // E.g., splitting dim 0 of a [5,2] tensor with split_size 2 would produce
  // three tensors with size [2,2], [2,2], [1,2].
  int64_t remainder; // The size of the split dimension of the final out tensor.
  if (split_size >= dim_size) {
    // Note that this also handles the case where split_size == 0, avoiding a
    // division by zero in the other branch. When dim_size == 0 && split_size ==
    // 0, core PyTorch expects 1 output element.
    ET_CHECK_MSG(
        out.size() == 1,
        "Unexpected out.size() %zu: should be 1 because split_size %" PRId64
        " >= input.size(%" PRId64 ") %zd",
        out.size(),
        split_size,
        dim,
        dim_size);
    remainder = dim_size;
  } else {
    int64_t expected_out_len = (dim_size + split_size - 1) / split_size;
    ET_CHECK_MSG(
        out.size() == expected_out_len,
        "Unexpected out.size() %zu: ceil(input.size(%" PRId64
        ")=%zd"
        " / split_size=%" PRId64 ") is %" PRId64,
        out.size(),
        dim,
        dim_size,
        split_size,
        expected_out_len);
    remainder = dim_size % split_size;
    if (remainder == 0) {
      remainder = split_size;
    }
  }

  // Validate each output.
  for (size_t i = 0; i < out.size(); ++i) {
    // All output dtypes must match the input type.
    ET_CHECK_MSG(
        out[i].scalar_type() == input.scalar_type(),
        "out[%zu] dtype %hhd != input dtype %hhd",
        i,
        out[i].scalar_type(),
        input.scalar_type());

    // All outputs must have the same number of dimensions as the input.
    ET_CHECK_MSG(
        out[i].dim() == input.dim(),
        "out[%zu] dim %zd != input dim %zd",
        i,
        out[i].dim(),
        input.dim());

    // Check the shape of the output.
    for (ssize_t d = 0; d < out[i].dim(); ++d) {
      if (d == dim) {
        // This is the split dimension, which may be different.
        if (i < out.size() - 1) {
          // All outputs except the final one: split dimension should be
          // split_size.
          ET_CHECK_MSG(
              out[i].size(d) == split_size,
              "out[%zu].size(%zd) %zd != split_size %" PRId64,
              i,
              d,
              out[i].size(d),
              split_size);
        } else {
          // The final output: split dimension should be the remainder of
          // split_size.
          ET_CHECK_MSG(
              out[i].size(d) == remainder,
              "out[%zu].size(%zd) %zd != remainder %" PRId64,
              i,
              d,
              out[i].size(d),
              remainder);
        }
      } else {
        // Non-split output dimensions must be the same as the input dimension.
        ET_CHECK_MSG(
            out[i].size(d) == input.size(d),
            "out[%zu].size(%zd) %zd != input.size(%zd) %zd",
            i,
            d,
            out[i].size(d),
            d,
            input.size(d));
      }
    }
  }
}

} // namespace

/**
 * Splits the tensor into chunks of size `split_size` along the specified
 * dimension.
 *
 * The last chunk will be smaller if the tensor size along the given dimension
 * dim is not evenly divisible by `split_size`.
 *
 * split_copy.Tensor_out(Tensor input, int split_size, int dim=0, *,
 * Tensor(a!)[] out) -> ()
 */
void split_copy_Tensor_out(
    RuntimeContext& context,
    const Tensor& input,
    int64_t split_size,
    int64_t dim,
    TensorList out) {
  (void)context;
  // Support python-style negative indexing.
  if (dim < 0) {
    dim += input.dim();
  }
  check_args(input, split_size, dim, out);

  const size_t leading_dims = getLeadingDims(input, dim);
  const size_t trailing_dims = getTrailingDims(input, dim);

  const size_t element_size = input.element_size();
  const size_t step = input.size(dim) * trailing_dims * element_size;

  const char* input_data = input.data_ptr<char>();
  for (size_t i = 0, e = out.size(); i < e; ++i) {
    size_t num_bytes = out[i].size(dim) * trailing_dims * element_size;
    if (num_bytes == 0) {
      continue;
    }

    const char* src = input_data;
    char* dest = out[i].data_ptr<char>();
    for (size_t j = 0; j < leading_dims; ++j) {
      memcpy(dest, src, num_bytes);
      src += step;
      dest += num_bytes;
    }
    input_data += num_bytes;
  }
}

} // namespace native
} // namespace executor
} // namespace torch
