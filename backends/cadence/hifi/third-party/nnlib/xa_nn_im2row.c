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
            const int8_t *__restrict__ slice_im =
                data_im + (h_im * width + w_im) * channels;
            int8_t *__restrict__ slice_col = data_col + i_col * channels_col +
                                             (_kh * kernel_w + _kw) * channels;
            // If the coordinates were within the input domain, we copy
            // 'channels' contiguous values. Otherwise we will fill the output
            // with 0's.
            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              const ae_int32x2 *pae_inp = (const ae_int32x2 *)slice_im;
              ae_int32x2 *pae_out = (ae_int32x2 *)slice_col;
              ae_valign inp_a, out_a;
              inp_a = AE_LA64_PP(pae_inp);
              out_a = AE_ZALIGN64();

              ae_int32x2 d0;
              for (int ic = 0; ic < channels >> 3; ic++) {
                AE_LA32X2_IP(d0, inp_a, pae_inp);
                AE_SA32X2_IP(d0, out_a, pae_out);
              }
              AE_SA64POS_FP(out_a, pae_out);

              int remainder = channels & 7;
              int8_t *ptmp_in = (int8_t *)pae_inp;
              int8_t *ptmp_out = (int8_t *)pae_out;
              for (int ic = 0; ic < remainder; ic++) {
                *ptmp_out++ = *ptmp_in++;
              }
            } else {
              for (int i = 0; i < channels; i++) {
                slice_col[i] = (int8_t)(in_zero_point);
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
