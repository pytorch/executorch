// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <algorithm>

#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace HiFi {
namespace native {

template <typename T>
__attribute__((always_inline)) void im2row_(
    const T* __restrict__ data_im,
    const int32_t in_zero_point,
    /* input parameters*/
    const int32_t channels,
    const int32_t height,
    const int32_t width,
    /* output parameters */
    const int32_t out_height,
    const int32_t out_width,
    /* convolution parameters */
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    T* __restrict__ data_col,
    bool channels_last) {
  // Consider convolving the input image of dimensions channels * height * width
  // (or height * width * channels for NHWC layout) with a filter of dimensions
  // channels * kernels_h * kernels_w. Assume that this convolution will produce
  // an output of dimensinos out_height x out_width. For each point the output,
  // im2row takes the data from the input that is used in the computation of
  // that output point, and flattens it into a vector of size channels_col =
  // channels * kernel_h * kernel_w. The output of im2row will therefore be a 2D
  // array of size (out_height * out_width) x channels_col
  const int32_t channels_col = channels * kernel_h * kernel_w;

  // If the layout is NHWC, we can copy 'channels' worth of contiguous data
  // points when performing im2row.
  if (channels_last) {
    // Iterate over the output domain
    for (int _h = 0; _h < out_height; ++_h) {
      for (int _w = 0; _w < out_width; ++_w) {
        int32_t i_col = _h * out_width + _w;
        // Each point in the output domain is the result of applying a filter of
        // size kernel_h x kernel_w x channels on the input. But since channels
        // is contiguous, we will not explicitly have a loop for it.
        for (int _kh = 0; _kh < kernel_h; ++_kh) {
          int32_t h_im = _h * stride_h - pad_h + _kh * dilation_h;
          for (int _kw = 0; _kw < kernel_w; ++_kw) {
            int32_t w_im = _w * stride_w - pad_w + _kw * dilation_w;

            // h_im and w_im are the actual height and width coordinates of the
            // input tensor from where we need to copy 'channels' points.
            const T* __restrict__ slice_im =
                data_im + (h_im * width + w_im) * channels;
            T* __restrict__ slice_col = data_col + i_col * channels_col +
                (_kh * kernel_w + _kw) * channels;
            // If the coordinates were within the input domain, we copy
            // 'channels' contiguous values. Otherwise we will fill the output
            // with 0's.
            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              std::memcpy(slice_col, slice_im, channels * sizeof(T));
            } else {
              std::fill_n(slice_col, channels, T(in_zero_point));
            }
          }
        }
      }
    }
  } else {
    // Iterate over the output domain
    for (int _h = 0; _h < out_height; ++_h) {
      for (int _w = 0; _w < out_width; ++_w) {
        int32_t i_col = _h * out_width + _w;

        // Each point in the output domain is the result of applying a filter
        // of size chanenls * kernel_h x kernel_w on the input
        for (int _c = 0; _c < channels; ++_c) {
          for (int _kh = 0; _kh < kernel_h; ++_kh) {
            for (int _kw = 0; _kw < kernel_w; ++_kw) {
              // c_col is the linearized access in the channels_col vector.
              int32_t c_col = (_c * kernel_h + _kh) * kernel_w + _kw;
              // h_im and w_im are the actual height and width coordinates of
              // the input tensor that we need to copy to the output.
              int32_t h_im = _h * stride_h - pad_h + _kh * dilation_h;
              int32_t w_im = _w * stride_w - pad_w + _kw * dilation_w;
              // If the current data access is within the input tensor, copy the
              // value
              data_col[i_col * channels_col + c_col] =
                  (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                  ? data_im[(_c * height + h_im) * width + w_im]
                  : static_cast<T>(in_zero_point);
            }
          }
        }
      }
    }
  }
}

void im2row_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& in_zero_point,
    bool channel_last,
    Tensor& out) {
  // Compute the input tensor's dims
  bool unit_height = input.dim() == 3;
  const int32_t batch_size = input.size(0);
  const int32_t in_c =
      channel_last ? input.size(3 - unit_height) : input.size(1);
  const int32_t in_h =
      unit_height ? 1 : (channel_last ? input.size(1) : input.size(2));
  const int32_t in_w =
      channel_last ? input.size(2 - unit_height) : input.size(3 - unit_height);

  // Get the kernel parameters
  int32_t kernel_h = kernel_size[0];
  int32_t kernel_w = kernel_size[1];
  int32_t dilation_h = dilation[0];
  int32_t dilation_w = dilation[1];
  int32_t pad_h = padding[0];
  int32_t pad_w = padding[1];
  int32_t stride_h = stride[0];
  int32_t stride_w = stride[1];

  // If we were to apply a convolution on the input tensor, compute the output
  // height and width.
  int32_t out_h =
      (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  int32_t out_w =
      (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

  ET_DCHECK_MSG(
      (out_h * out_w) == out.size(1), "dimension mismatch for output");
  ET_DCHECK_MSG(
      (kernel_h * kernel_w * in_c) == out.size(2),
      "dimension mismatch for output");
  // Check if the input is per-tensor quantized or per-channel quantized. The
  // zero point for each batch could differ for per-channel quantized input.
  bool per_tensor_quantized = in_zero_point.numel() == 1;

  bool optimized = false;
  if (input.scalar_type() == ScalarType::Char ||
      input.scalar_type() == ScalarType::Byte)
    optimized = true;

  if (optimized) {
    const int8_t* __restrict__ in_data =
        (WORD8* __restrict__)input.const_data_ptr<int8_t>();
    int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();
    const int32_t* __restrict__ zero_point =
        in_zero_point.const_data_ptr<int32_t>();
    int32_t in_plane = in_c * in_h * in_w;
    int32_t out_plane = kernel_h * kernel_w * in_c * out_h * out_w;
    for (size_t n = 0; n < batch_size; ++n) {
      xa_nn_im2row_quantized(
          &in_data[n * in_plane],
          per_tensor_quantized ? zero_point[0] : zero_point[n],
          in_c,
          in_h,
          in_w,
          out_h,
          out_w,
          kernel_h,
          kernel_w,
          pad_h,
          pad_w,
          stride_h,
          stride_w,
          dilation_h,
          dilation_w,
          &out_data[n * out_plane],
          channel_last ? 1 : 0);
    }
  } else {
#define typed_im2row(dtype, ctype)                                     \
  case ScalarType::dtype: {                                            \
    const ctype* __restrict__ in_data = input.const_data_ptr<ctype>(); \
    ctype* __restrict__ out_data = out.mutable_data_ptr<ctype>();      \
    const int32_t* __restrict__ zero_point =                           \
        in_zero_point.const_data_ptr<int32_t>();                       \
    int32_t in_plane = in_c * in_h * in_w;                             \
    int32_t out_plane = kernel_h * kernel_w * in_c * out_h * out_w;    \
    for (size_t n = 0; n < batch_size; ++n) {                          \
      im2row_<ctype>(                                                  \
          &in_data[n * in_plane],                                      \
          per_tensor_quantized ? zero_point[0] : zero_point[n],        \
          in_c,                                                        \
          in_h,                                                        \
          in_w,                                                        \
          out_h,                                                       \
          out_w,                                                       \
          kernel_h,                                                    \
          kernel_w,                                                    \
          pad_h,                                                       \
          pad_w,                                                       \
          stride_h,                                                    \
          stride_w,                                                    \
          dilation_h,                                                  \
          dilation_w,                                                  \
          &out_data[n * out_plane],                                    \
          channel_last);                                               \
    }                                                                  \
    break;                                                             \
  }

    ScalarType dtype = input.scalar_type();
    switch (dtype) {
      typed_im2row(Float, float);
      typed_im2row(Byte, uint8_t);
      typed_im2row(Char, int8_t);
      default:
        ET_DCHECK_MSG(
            false,
            "im2row not implemented for dtype %s",
            torch::executor::toString(dtype));
    }
#undef typed_im2row
  }
}

void im2row_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    int64_t in_zero_point,
    bool channel_last,
    Tensor& out) {
  // Compute the input tensor's dims
  bool unit_height = input.dim() == 3;
  const int32_t batch_size = input.size(0);
  const int32_t in_c =
      channel_last ? input.size(3 - unit_height) : input.size(1);
  const int32_t in_h =
      unit_height ? 1 : (channel_last ? input.size(1) : input.size(2));
  const int32_t in_w =
      channel_last ? input.size(2 - unit_height) : input.size(3 - unit_height);

  // Get the kernel parameters
  int32_t kernel_h = kernel_size[0];
  int32_t kernel_w = kernel_size[1];
  int32_t dilation_h = dilation[0];
  int32_t dilation_w = dilation[1];
  int32_t pad_h = padding[0];
  int32_t pad_w = padding[1];
  int32_t stride_h = stride[0];
  int32_t stride_w = stride[1];

  // If we were to apply a convolution on the input tensor, compute the output
  // height and width.
  int32_t out_h =
      (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  int32_t out_w =
      (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

  ET_DCHECK_MSG(
      (out_h * out_w) == out.size(1), "dimension mismatch for output");
  ET_DCHECK_MSG(
      (kernel_h * kernel_w * in_c) == out.size(2),
      "dimension mismatch for output");

  bool optimized = false;
  if (input.scalar_type() == ScalarType::Char ||
      input.scalar_type() == ScalarType::Byte)
    optimized = true;

  if (optimized) {
    const int8_t* __restrict__ in_data =
        (WORD8* __restrict__)input.const_data_ptr<int8_t>();
    int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();
    int32_t in_plane = in_c * in_h * in_w;
    int32_t out_plane = kernel_h * kernel_w * in_c * out_h * out_w;

    for (size_t n = 0; n < batch_size; ++n) {
      xa_nn_im2row_quantized(
          &in_data[n * in_plane],
          (int32_t)in_zero_point,
          in_c,
          in_h,
          in_w,
          out_h,
          out_w,
          kernel_h,
          kernel_w,
          pad_h,
          pad_w,
          stride_h,
          stride_w,
          dilation_h,
          dilation_w,
          &out_data[n * out_plane],
          channel_last ? 1 : 0);
    }
  } else {
#define typed_im2row_per_tensor(dtype, ctype)                          \
  case ScalarType::dtype: {                                            \
    const ctype* __restrict__ in_data = input.const_data_ptr<ctype>(); \
    ctype* __restrict__ out_data = out.mutable_data_ptr<ctype>();      \
    int32_t in_plane = in_c * in_h * in_w;                             \
    int32_t out_plane = kernel_h * kernel_w * in_c * out_h * out_w;    \
    for (size_t n = 0; n < batch_size; ++n) {                          \
      im2row_<ctype>(                                                  \
          &in_data[n * in_plane],                                      \
          in_zero_point,                                               \
          in_c,                                                        \
          in_h,                                                        \
          in_w,                                                        \
          out_h,                                                       \
          out_w,                                                       \
          kernel_h,                                                    \
          kernel_w,                                                    \
          pad_h,                                                       \
          pad_w,                                                       \
          stride_h,                                                    \
          stride_w,                                                    \
          dilation_h,                                                  \
          dilation_w,                                                  \
          &out_data[n * out_plane],                                    \
          channel_last);                                               \
    }                                                                  \
    break;                                                             \
  }

    ScalarType dtype = input.scalar_type();
    switch (dtype) {
      typed_im2row_per_tensor(Float, float);
      typed_im2row_per_tensor(Byte, uint8_t);
      typed_im2row_per_tensor(Char, int8_t);
      default:
        ET_DCHECK_MSG(
            false,
            "im2row.per_tensor not implemented for dtype %s",
            torch::executor::toString(dtype));
    }
#undef typed_im2row_per_tensor
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
