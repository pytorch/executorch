// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/kernels/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

// TODO(gasoonjia): Move this to a common spot so all implementation of
// this operator can share it. (e.g., DSP-specific)
/// Asserts that the parameters are valid.
void check_stack_out_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Stack expects non-empty tensor list
  ET_CHECK_MSG(tensors.size() > 0, "Stack expects non-empty tensor list");

  // Ensure dim is in range. Use `out` as a proxy for all input tensors, since
  // they will all need to have the same number of dimensions besides the dim
  // one.
  ET_CHECK_MSG(
      dim >= 0 && dim < out.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      out.dim());

  for (size_t i = 0; i < tensors.size(); i++) {
    // All input dtypes must match the output dtype.
    ET_CHECK_MSG(
        tensors[i].scalar_type() == out.scalar_type(),
        "tensors[%zu] dtype %hhd != out dtype %hhd",
        i,
        tensors[i].scalar_type(),
        out.scalar_type());

    // All input tensors need to be of the same size
    // Also, since we create a new axis in output for stacking, the output.dim()
    // should be one larger than input.dim()
    // https://pytorch.org/docs/stable/generated/torch.stack.html
    ET_CHECK_MSG(
        tensors[i].dim() == out.dim() - 1,
        "tensors[%zu].dim() %zd != out.dim() - 1 %zd",
        i,
        tensors[i].dim(),
        out.dim() - 1);

    // The size of each input tensor should be the same. Here we use `out` as
    // proxy for comparsion. Also, the size of output tensor should follow these
    // rules:
    // - For any input tensor, its size(i) == output.size(i) if i < dim, and its
    //   size(i) == output.size(i+1) if i >= dim
    // - For the cat dimension (output[dim]), its size should be the number of
    //   input tensors
    for (size_t d = 0; d < tensors[i].dim(); d++) {
      if (d < dim) {
        ET_CHECK_MSG(
            tensors[i].size(d) == out.size(d),
            "tensors[%zu].size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
            i,
            d,
            tensors[i].size(d),
            d,
            out.size(d),
            dim);
      } else {
        ET_CHECK_MSG(
            tensors[i].size(d) == out.size(d + 1),
            "tensors[%zu].size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
            i,
            d,
            tensors[i].size(d),
            d + 1,
            out.size(d + 1),
            dim);
      }
    }
  }

  // The size of the stack dimension of the output should be the number of
  // input tensors
  ET_CHECK_MSG(
      out.size(dim) == tensors.size(),
      "out.size(%" PRId64 ") %zd != number of input tensors %zu",
      dim,
      out.size(dim),
      tensors.size());
}
} // namespace

/// stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor& stack_out(
    RuntimeContext& context,
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)context;
  // Support python-style negative indexing. E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.
  if (dim < 0) {
    dim += out.dim();
  }

  // Assert that the args are valid.
  check_stack_out_args(tensors, dim, out);

  // If one tensor is empty tensor, all tensors are empty since they share same
  // size. Under that, no need do anything. Just return the out.
  if (tensors[0].numel() == 0) {
    return out;
  }

  size_t leading_dim = getLeadingDims(out, dim);
  size_t trailing_dim = getTrailingDims(out, dim);
  size_t num_of_tensors = tensors.size();

  size_t chunk_size = trailing_dim * out.element_size();

  char* dst_ptr = out.data_ptr<char>();

  for (int i = 0; i < leading_dim; i++) {
    for (int j = 0; j < num_of_tensors; j++) {
      char* src_ptr = tensors[j].data_ptr<char>() + chunk_size * i;
      memcpy(dst_ptr, src_ptr, chunk_size);
      dst_ptr += chunk_size;
    }
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
