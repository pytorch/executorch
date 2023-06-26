// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

/**
 * Assumptions for inputs:
 * 1. index is 1-D tensor and monotonically increasing
 * 2. output size is the same as input size except for `dim`. output.size(dim)
 * == index.numel().
 */
void check_index_select_args(
    const Tensor& input,
    int64_t dim,
    const Tensor& index,
    Tensor& output) {
  // Check dim. The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  // Input output should have the same dim
  ET_CHECK_MSG(
      input.dim() == output.dim(),
      "input.dim() %zd not equal to output.dim() %zd",
      ssize_t(input.dim()),
      ssize_t(output.dim()));

  // Input dtype shall match the output dtype.
  ET_CHECK_SAME_DTYPE2(input, output);

  // output.numel() needs to follow some constraints.
  size_t trailing_dims = getTrailingDims(input, dim);
  size_t leading_dims = getLeadingDims(input, dim);
  ET_CHECK_MSG(
      output.numel() == (leading_dims * index.numel() * trailing_dims),
      "output.numel() %zd != (leading_dims %zd * index.numel() %zd * trailing_dims %zd)",
      output.numel(),
      leading_dims,
      index.numel(),
      trailing_dims);

  // Index should be a 1-D LongTensor, check if any index is out of bound
  ET_CHECK_MSG(
      index.scalar_type() == ScalarType::Long, "index scalar_type not long");
  ET_CHECK_MSG(index.dim() == 1, "index.dim() %zd != 1", index.dim());

  const int64_t* src = index.data_ptr<int64_t>();
  for (auto i = 1; i < index.numel(); i++) {
    ET_CHECK_MSG(
        src[i] >= 0 && src[i] < input.size(dim),
        "index[%d] %" PRId64 " is out of bound [0, input.size(dim) %zd)",
        i,
        src[i],
        ssize_t(input.size(dim)));
  }
  // The sizes of output should match input, except at dim where the size should
  // be equal to index.numel().
  for (auto i = 0; i < output.dim(); i++) {
    if (i == dim) {
      ET_CHECK_MSG(
          output.size(i) == index.numel(),
          "At dim %d, output.size(dim) %zd != index.numel() %zd",
          i,
          output.size(i),
          index.numel());
    } else {
      ET_CHECK_MSG(
          output.size(i) == input.size(i),
          "At dim %d, output.size(dim) %zd != input.size(dim) %zd",
          i,
          output.size(i),
          input.size(i));
    }
  }
}

} // namespace

/// aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!)
/// out) -> Tensor(a!)
Tensor& index_select_out(
    RuntimeContext& context,
    const Tensor& input,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  if (dim < 0) {
    dim += input.dim();
  }

  Tensor::SizesType expected_output_size[16];
  for (size_t i = 0; i < out.dim(); ++i) {
    if (i != dim) {
      expected_output_size[i] = input.size(i);
    } else {
      expected_output_size[i] = index.numel();
    }
  }
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(out.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  check_index_select_args(input, dim, index, out);
  size_t out_dim_length = out.size(dim);
  size_t in_dim_length = input.size(dim);

  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_dims = getTrailingDims(input, dim);

  size_t length_per_step = trailing_dims * input.element_size();

  const char* input_data = input.data_ptr<char>();
  char* out_data = out.data_ptr<char>();
  const int64_t* index_arr = index.data_ptr<int64_t>();
  for (int i = 0; i < leading_dims; i++) {
    const char* src = input_data + i * in_dim_length * length_per_step;
    char* dest = out_data + i * out_dim_length * length_per_step;
    for (auto j = 0; j < out_dim_length; j++) {
      const char* copy_src = src + index_arr[j] * length_per_step;
      memcpy(dest, copy_src, length_per_step);
      dest += length_per_step;
    }
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
