/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

namespace {

bool check_max_pool2d_backward_args(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    const Tensor& grad_input) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_output, input));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_input, input));

  ET_CHECK_OR_RETURN_FALSE(
      check_max_pool2d_with_indices_args(
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          grad_output,
          indices),
      "Invalid max_pool_2d arguments");

  size_t output_ndim = 0;
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_max_pool2d_with_indices_out_target_size(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output_sizes,
      &output_ndim);

  ET_LOG_AND_RETURN_IF_FALSE(
      output_size_is_valid({output_sizes, output_ndim}, 2));

  ET_CHECK_OR_RETURN_FALSE(
      grad_output.dim() == input.dim(),
      "grad_output should have same number of dimensions as input; grad_output.dim() = %" ET_PRI_TENSOR_DIM
      ", input.dim() = %" ET_PRI_TENSOR_DIM,
      grad_output.dim(),
      input.dim());

  ET_LOG_AND_RETURN_IF_FALSE(
      tensor_has_expected_size(grad_output, {output_sizes, output_ndim}));

  return true;
}

template <typename CTYPE, bool is_3d>
void max_pool_backward_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  const CTYPE* grad_output_data = grad_output.const_data_ptr<CTYPE>();
  const int64_t* indices_data = indices.const_data_ptr<int64_t>();
  CTYPE* grad_input_data = grad_input.mutable_data_ptr<CTYPE>();

  // treat batch size and channels as one dimension
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  int64_t ndim = grad_output.dim();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? grad_output.size(0)
                         : grad_output.size(0) * grad_output.size(1);
  } else {
    channels = ndim == 3 ? grad_output.size(0)
                         : grad_output.size(0) * grad_output.size(1);
  }
  int64_t input_depth = is_3d ? grad_input.size(-3) : 1;

  int64_t input_height = grad_input.size(ndim - 2);
  int64_t input_width = grad_input.size(ndim - 1);
  int64_t output_depth = is_3d ? grad_output.size(ndim - 3) : 1;
  int64_t output_height = grad_output.size(ndim - 2);
  int64_t output_width = grad_output.size(ndim - 1);

  for (int64_t c = 0; c < channels; ++c) {
    CTYPE* grad_input_ptr =
        grad_input_data + c * input_depth * input_height * input_width;
    const CTYPE* grad_output_ptr =
        grad_output_data + c * output_depth * output_height * output_width;
    const int64_t* indices_ptr =
        indices_data + c * output_depth * output_height * output_width;

    for (int64_t od = 0; od < output_depth; od++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          // retrieve position of max
          int64_t index =
              od * output_height * output_width + oh * output_width + ow;
          int64_t maxindex = indices_ptr[index];
          if (maxindex != -1) {
            // update gradient
            grad_input_ptr[maxindex] += grad_output_ptr[index];
          }
        }
      }
    }
  }
}

} // namespace

Tensor& max_pool2d_with_indices_backward_out(
    KernelRuntimeContext& ctx,
    const Tensor& grad_output,
    const Tensor& input,
    ET_UNUSED IntArrayRef kernel_size,
    ET_UNUSED IntArrayRef stride,
    ET_UNUSED IntArrayRef padding,
    ET_UNUSED IntArrayRef dilation,
    ET_UNUSED bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_max_pool2d_backward_args(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices,
          grad_input),
      InvalidArgument,
      grad_input);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(grad_input, input.sizes()) == Error::Ok,
      InvalidArgument,
      grad_input);

  static constexpr auto name = "max_pool2d_with_indices_backward.grad_input";

  ET_SWITCH_FLOATHBF16_TYPES(input.scalar_type(), ctx, name, CTYPE, [&]() {
    max_pool_backward_impl<CTYPE, false>(grad_input, grad_output, indices);
  });

  return grad_input;
}

} // namespace native
} // namespace executor
} // namespace torch
