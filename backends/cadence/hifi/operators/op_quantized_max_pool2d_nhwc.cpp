/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

Tensor& quantized_max_pool2d_nhwc_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output) {
  // NHWC layout: [N, H, W, C]
  const int32_t batch_size = input.size(0);
  const int32_t in_height = input.size(1);
  const int32_t in_width = input.size(2);
  const int32_t channels = input.size(3);

  const int32_t out_height = output.size(1);
  const int32_t out_width = output.size(2);

  const int32_t kernel_h = kernel_size[0];
  const int32_t kernel_w = kernel_size[1];
  const int32_t stride_h = stride[0];
  const int32_t stride_w = stride[1];
  const int32_t pad_h = padding[0];
  const int32_t pad_w = padding[1];

  // Determine NNLIB precision constants based on dtype
  ScalarType dtype = input.scalar_type();
  int32_t nnlib_precision;
  switch (dtype) {
    case ScalarType::Char: // int8
      nnlib_precision = PREC_SYM8S;
      break;
    case ScalarType::Byte: // uint8
      nnlib_precision = PREC_ASYM8U;
      break;
    default:
      ET_DCHECK_MSG(
          false,
          "Unsupported dtype %s for HiFi quantized_max_pool2d_nhwc",
          torch::executor::toString(dtype));
      return output;
  }

  // Compute scratch buffer size for NNLIB maxpool
  int32_t scratch_size = xa_nn_maxpool_getsize(
      channels,
      nnlib_precision,
      nnlib_precision,
      in_height,
      in_width,
      kernel_h,
      kernel_w,
      stride_w, // x_stride
      stride_h, // y_stride
      pad_w, // x_padding
      pad_h, // y_padding
      out_height,
      out_width,
      0, // inp_data_format: 0 = NHWC
      0); // out_data_format: 0 = NHWC
  ET_DCHECK_MSG(scratch_size >= 0, "xa_nn_maxpool_getsize failed");

  // Allocate aligned scratch memory
  void* p_scratch = kernels::allocate_temp_memory(ctx, scratch_size);

  // Process each batch using NNLIB optimized maxpool kernel
  for (int32_t n = 0; n < batch_size; ++n) {
    const int32_t spatial_size = in_height * in_width * channels;
    const int32_t out_spatial_size = out_height * out_width * channels;

    int32_t ret;
    if (dtype == ScalarType::Char) {
      const int8_t* in_batch =
          input.const_data_ptr<int8_t>() + n * spatial_size;
      int8_t* out_batch =
          output.mutable_data_ptr<int8_t>() + n * out_spatial_size;

      ret = xa_nn_maxpool_8(
          out_batch,
          in_batch,
          in_height,
          in_width,
          channels,
          kernel_h,
          kernel_w,
          stride_w, // x_stride
          stride_h, // y_stride
          pad_w, // x_padding
          pad_h, // y_padding
          out_height,
          out_width,
          0, // inp_data_format: NHWC
          0, // out_data_format: NHWC
          p_scratch);
    } else {
      const uint8_t* in_batch =
          input.const_data_ptr<uint8_t>() + n * spatial_size;
      uint8_t* out_batch =
          output.mutable_data_ptr<uint8_t>() + n * out_spatial_size;

      ret = xa_nn_maxpool_asym8(
          out_batch,
          in_batch,
          in_height,
          in_width,
          channels,
          kernel_h,
          kernel_w,
          stride_w, // x_stride
          stride_h, // y_stride
          pad_w, // x_padding
          pad_h, // y_padding
          out_height,
          out_width,
          0, // inp_data_format: NHWC
          0, // out_data_format: NHWC
          p_scratch);
    }
    ET_DCHECK_MSG(ret == 0, "HiFi xa_nn_maxpool failed");
  }

  return output;
}

} // namespace native
} // namespace HiFi
} // namespace impl
