#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_err_chk.h"
#include "xa_type_def.h"
// #include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"

WORD32 xa_nn_im2row_quantized(
    const WORD8 *__restrict__ data_im, const WORD32 in_zero_point,
    /* input parameters*/
    const WORD32 channels, const WORD32 height, const WORD32 width,
    /* output parameters */
    const WORD32 out_height, const WORD32 out_width,
    /* convolution parameters */
    const WORD32 kernel_h, const WORD32 kernel_w, const WORD32 pad_h,
    const WORD32 pad_w, const WORD32 stride_h, const WORD32 stride_w,
    const WORD32 dilation_h, const WORD32 dilation_w,
    WORD8 *__restrict__ data_col, WORD32 channels_last) {
  const WORD32 channels_col = channels * kernel_h * kernel_w;

  // If the layout is NHWC, the input data is contiguous per-pixel (H, W, C).
  // The output layout must match torch.nn.functional.unfold, which is [c][kp]:
  //   output[c * num_kp + kp] for each output position.
  if (channels_last) {
    const int32_t num_kp = kernel_h * kernel_w;
    // Iterate over the output domain
    for (int _h = 0; _h < out_height; ++_h) {
      for (int _w = 0; _w < out_width; ++_w) {
        int32_t i_col = _h * out_width + _w;
        for (int _kh = 0; _kh < kernel_h; ++_kh) {
          int32_t h_im = _h * stride_h - pad_h + _kh * dilation_h;
          for (int _kw = 0; _kw < kernel_w; ++_kw) {
            int32_t w_im = _w * stride_w - pad_w + _kw * dilation_w;
            int32_t kp = _kh * kernel_w + _kw;

            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              const int8_t *__restrict__ pixel =
                  data_im + (h_im * width + w_im) * channels;
              for (int _c = 0; _c < channels; ++_c) {
                data_col[i_col * channels_col + _c * num_kp + kp] = pixel[_c];
              }
            } else {
              for (int _c = 0; _c < channels; ++_c) {
                data_col[i_col * channels_col + _c * num_kp + kp] =
                    (int8_t)(in_zero_point);
              }
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
              if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                data_col[i_col * channels_col + c_col] =
                    data_im[(_c * height + h_im) * width + w_im];
              else
                data_col[i_col * channels_col + c_col] = (int8_t)in_zero_point;
            }
          }
        }
      }
    }
  }

  return 0;
}
