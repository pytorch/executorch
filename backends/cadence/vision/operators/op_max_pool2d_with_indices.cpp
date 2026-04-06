/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <dump_tensor.h>
#include <cstring>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::Tensor;
using executorch::aten::ScalarType;
using executorch::aten::IntArrayRef;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace impl {
namespace vision {
namespace native {

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices) {
  TIME_DECL(maxpool);
  TIME_START(maxpool);

  std::tuple<Tensor&, Tensor&> ret_val(out, indices);

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_max_pool2d_with_indices_args(
          in, kernel_size, stride, padding, dilation, ceil_mode, out, indices),
      InvalidArgument,
      ret_val);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[executorch::runtime::kTensorDimensionLimit];
  torch::executor::get_max_pool2d_with_indices_out_target_size(
      in,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output_sizes,
      &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::output_size_is_valid({output_sizes, output_ndim}, 2),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(indices, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  bool optimized = false;

  if (stride[0] == 2 && stride[1] == 2)
    optimized = true;

  if (optimized){
    float32_t *ptr_out = (float32_t *) out.const_data_ptr<float>();
    const float32_t *ptr_inp = (float32_t *) in.const_data_ptr<float>();
    int batch = in.size(0);
    int channels = in.size(1);
    int inp_height = in.size(2); int inp_width = in.size(3);
    int out_height = out.size(2); int out_width = out.size(3);
    int padded_height = inp_height + 2 * padding[0];
    int padded_width = inp_width + 2 * padding[1];
    int out_pitch_height = out_height; int out_pitch_width = out_width;
    uint8_t kernel_height = kernel_size[0];
    uint8_t kernel_width = kernel_size[1];

    // Allocate padded buffer
    size_t padded_size = batch * channels * padded_height * padded_width;
    executorch::runtime::Result<void*> temp_mem_res = 
        ctx.allocate_temp(padded_size * sizeof(float));
    float* padded_data = (float*)(temp_mem_res.ok() ? temp_mem_res.get() : nullptr);

    
    ET_KERNEL_CHECK(ctx, padded_data != nullptr, MemoryAllocationFailed, ret_val);
    
    // Invalidate input cache: previous op (dequantize) may have written via DMA,
    // bypassing cache. CPU memcpy below would read stale cache lines.
    xthal_dcache_region_invalidate((void*)ptr_inp, sizeof(float) * in.numel());
    
    // Initialize entire buffer with MIN_FLT32
    std::fill_n(padded_data, padded_size, MIN_FLT32);
    
    // Copy input data with padding
    for (int bc = 0; bc < batch * channels; bc++) {
      const float* src = ptr_inp + bc * inp_height * inp_width;
      float* dst = padded_data + bc * padded_height * padded_width + padding[0] * padded_width + padding[1];
      
      for (int h = 0; h < inp_height; h++) {
        std::memcpy(dst, src, inp_width * sizeof(float));
        src += inp_width;
        dst += padded_width;
      }
    }

    // Writeback padded buffer from cache to system memory so HW kernel sees it
    xthal_dcache_region_writeback(padded_data, sizeof(float) * padded_size);

    // Process each batch*channel plane independently
    for (int bc = 0; bc < batch * channels; bc++) {
      float32_t* plane_out = ptr_out + bc * out_height * out_width;
      const float32_t* plane_inp = padded_data + bc * padded_height * padded_width;

      maxpool2d_j2x2_f32(plane_out, plane_inp, inp_height, inp_width,
          out_height, out_width, padded_width, padded_height,
          out_pitch_width, out_pitch_height, kernel_height, kernel_width);
    }

    // Invalidate output cache lines: HW kernel wrote to system memory,
    // CPU/next-op must not see stale cache
    xthal_dcache_region_invalidate(ptr_out, sizeof(float) * out.numel());

    TIME_END(maxpool);
    TIME_DISPLAY(maxpool, in.numel(), "floats");
    DUMP_TENSOR(max_pool2d, out);

    return ret_val;
  }

  ScalarType in_type = in.scalar_type();
  ET_SWITCH_REALHBF16_TYPES(
      in_type, ctx, "max_pool2d_with_indices.out", CTYPE, [&]() {
      torch::executor::apply_kernel_2d_reduce_then_map_fn<CTYPE>(
            [](const CTYPE in_val,
               const int64_t in_idx,
               const CTYPE accum,
               const int64_t accum_idx) {
              if (in_val > accum) {
                return std::tuple<CTYPE, int64_t>(in_val, in_idx);
              }
              return std::tuple<CTYPE, int64_t>(accum, accum_idx);
            },
            // Max pooling does not need to post-process the accumulated output
            [](const int64_t count, const CTYPE accum) { return accum; },
            /*include_pad=*/false,
            in,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
            {indices});
      });

  TIME_END(maxpool);
  TIME_DISPLAY(maxpool, in.numel(), "floats");
  DUMP_TENSOR(max_pool2d, out);

  return ret_val;
}

} // namespace native
} // namespace vision
} // namespace impl
