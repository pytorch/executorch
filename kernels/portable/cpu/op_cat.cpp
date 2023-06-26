// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/kernels/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

// TODO(T128954939): Move this to a common spot so all implementation of
// this operator can share it. (e.g., DSP-specific)
/// Asserts that the parameters are valid.
void check_cat_out_args(
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Ensure the input tensors list is non-empty
  ET_CHECK_MSG(tensors.size() > 0, "Cat expects non-empty tensor list");

  // Ensure dim is in range. Use `out` as a proxy for all input tensors, since
  // they will all need to have the same number of dimensions.
  ET_CHECK_MSG(
      dim >= 0 && dim < out.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      out.dim());

  size_t cat_dim_size = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    // All input dtypes must match the output dtype.
    ET_CHECK_MSG(
        tensors[i].scalar_type() == out.scalar_type(),
        "tensors[%zu] dtype %hhd != out dtype %hhd",
        i,
        tensors[i].scalar_type(),
        out.scalar_type());

    // Empty tensors have no shape constraints.
    if (tensors[i].numel() == 0) {
      continue;
    }

    // All input tensors must have the same number of dimensions as the output.
    ET_CHECK_MSG(
        tensors[i].dim() == out.dim(),
        "tensors[%zu].dim() %zd != out.dim() %zd",
        i,
        tensors[i].dim(),
        out.dim());

    // "All tensors must either have the same shape (except in the concatenating
    // dimension) or be empty."
    // https://pytorch.org/docs/stable/generated/torch.cat.html
    for (size_t d = 0; d < tensors[i].dim(); ++d) {
      if (d != dim) {
        ET_CHECK_MSG(
            tensors[i].size(d) == out.size(d),
            "tensors[%zu].size(%zu) %zd != out.size(%zu) %zd",
            i,
            d,
            tensors[i].size(d),
            d,
            out.size(d));
      }
    }

    cat_dim_size += tensors[i].size(dim);
  }

  // The size of the cat dimension of the output should be the sum of the
  // input cat dimension sizes.
  ET_CHECK_MSG(
      out.size(dim) == cat_dim_size,
      "out.size(%" PRId64 ") %zd != %zu",
      dim,
      out.size(dim),
      cat_dim_size);
}

void resize_out_tensor(
    exec_aten::ArrayRef<Tensor>& tensors,
    int64_t dim,
    Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];

  // Some elements of expected_output_size may not be set during the loop
  // over all the tensors. Set all of them ahead of time here so that none are
  // unset by the end of that loop
  for (size_t i = 0; i < out.dim(); ++i) {
    expected_output_size[i] = out.size(i);
  }

  size_t cat_dim_size = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    // Empty tensors have no shape constraints.
    if (tensors[i].numel() == 0) {
      continue;
    }
    for (size_t d = 0; d < tensors[i].dim(); ++d) {
      if (d != dim) {
        expected_output_size[d] = tensors[i].size(d);
      }
    }
    cat_dim_size += tensors[i].size(dim);
  }

  expected_output_size[dim] = cat_dim_size;

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in cat_out");
}
} // namespace

/// cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor& cat_out(
    RuntimeContext& context,
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  // Support python-style negative indexing. E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.
  if (dim < 0) {
    dim += out.dim();
  }

  resize_out_tensor(tensors, dim, out);

  // Assert that the args are valid.
  check_cat_out_args(tensors, dim, out);

  size_t cat_dim = out.size(dim);

  size_t leading_dims = getLeadingDims(out, dim);
  size_t trailing_dims = getTrailingDims(out, dim);

  size_t element_size = out.element_size();
  size_t step = cat_dim * trailing_dims * element_size;

  char* out_data = out.data_ptr<char>();
  for (size_t i = 0, e = tensors.size(); i < e; ++i) {
    if (tensors[i].numel() == 0) {
      // Ignore empty tensor.
      continue;
    }
    size_t num_bytes = tensors[i].size(dim) * trailing_dims * element_size;

    const char* src = tensors[i].data_ptr<char>();
    char* dest = out_data;
    for (size_t j = 0; j < leading_dims; ++j) {
      memcpy(dest, src, num_bytes);
      dest += step;
      src += num_bytes;
    }
    out_data += num_bytes;
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
