/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <lib.h>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/backends/cadence/vision/operators/conv/conv_layer_configs.h>
#include <stdio.h>

// Forward declaration of conv_execute_kernel (defined in conv_kernel_dispatcher.c)
extern "C" {
typedef int XAI_ERR_TYPE;
XAI_ERR_TYPE conv_execute_kernel(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);
}


using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace vision {
namespace native {

// This implements a generic 2d conv kernel that operates on raw pointers.
// The version handles both quantized and fp32 convolutions.
// The input is of shape [n x c x h x w]
// The weight is of shape [oc x wc x wh x ww], where wc == c
// The output is of shape [n x oc x oh x ow]
// The bias is of shape [oc]
template <
    typename IT = float,
    typename WT = IT,
    typename BT = IT,
    typename OT = IT,
    bool quantized = false>
__attribute__((noinline)) void conv2d_nchw_core_generic(
    // All the arrays
    const IT* __restrict__ p_in,
    const WT* __restrict__ p_weight,
    const BT* __restrict__ p_bias,
    OT* __restrict__ p_out,
    // The array sizes
    int32_t n,
    int32_t c,
    int32_t h,
    int32_t w,
    int32_t oc,
    int32_t wc,
    int32_t wh,
    int32_t ww,
    int32_t oh,
    int32_t ow,
    // Stride
    int16_t s0,
    int16_t s1,
    // Padding
    int16_t p0,
    int16_t p1,
    // Dilation
    int16_t d0,
    int16_t d1,
    // Group for depthwise conv
    int16_t groups,
    // Optional args that are only relevant for quantized convolution
    // input zero point
    IT in_zero_point = 0,
    // weight zero point
    int32_t weight_zero_point = 0,
    float bias_scale = 1,
    float out_scale = 1,
    OT out_zero_point = 0) {
  float inv_out_scale = 1. / out_scale;
  bool zero_pad_unit_dilation = d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0;

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * c * h * w;
    OT* out_batch = p_out + _n * oc * oh * ow;
    // Compute separable convolution for each group
    for (int _g = 0; _g < groups; ++_g) {
      // Identify the input and output channels involved in the computation
      // of this group
      int sic = _g * icpg;
      int soc = _g * ocpg;
      // Populate all the output channels in the group
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        OT* out_plane = out_batch + _oc * oh * ow;
        const WT* weight_batch = p_weight + _oc * wc * wh * ww;
        // We compute one output channel at a time. The computation can be
        // thought of as a stencil computation: we iterate over an input of size
        // icpg x h x w, with a stencil of size icpg x wh x ww, to compute an
        // output channel of size 1 x oh x ow.
        for (int _h = 0, _oh = 0; _oh < oh; _h += s0, ++_oh) {
          for (int _w = 0, _ow = 0; _ow < ow; _w += s1, ++_ow) {
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to the
            // output channel being computed) with the corresponding weight
            // channel.
            // If the padding is 0, and dilation is 1, then we can remove the
            // unnecessary checks, and simplify the code so that it can be
            // vectorized by Tensilica compiler.

            if (zero_pad_unit_dilation) {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int ioff = (_h + _wh) * w + (_w + _ww);
                    int woff = _wh * ww + _ww;
                    float lhs = in_plane[ioff] - in_zero_point;
                    float rhs = weight_plane[woff] -
                        (quantized ? weight_zero_point : 0);
                    acc += lhs * rhs;
                  }
                }
              }
            } else {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    if (((_h + d0 * _wh - p0) >= 0) &&
                        ((_h + d0 * _wh - p0) < h) &&
                        ((_w + d1 * _ww - p1) >= 0) &&
                        ((_w + d1 * _ww - p1) < w)) {
                      int ioff =
                          (_h + d0 * _wh - p0) * w + (_w + d1 * _ww - p1);
                      int woff = _wh * ww + _ww;
                      float lhs = in_plane[ioff] - in_zero_point;
                      float rhs = weight_plane[woff] -
                          (quantized ? weight_zero_point : 0);
                      acc += lhs * rhs;
                    }
                  }
                }
              }
            }
            if (quantized) {
              float val = bias_scale * acc;
              out_plane[_oh * ow + _ow] =
                  ::impl::generic::kernels::quantize<OT>(
                      val, inv_out_scale, out_zero_point);
            } else {
              out_plane[_oh * ow + _ow] = acc;
            }
          }
        }
      }
    }
  }
}

// The quantized convolution kernel. in_scale and weight_scale are implicit in
// bias_scale, since it is a product of the two. The kernel will branch to
// quantized::conv1d or quantized::conv2d based on the dimensionality of
// activation tensor.
void quantized_conv2d_nchw(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    int32_t in_zero_point,
    int32_t weight_zero_point,
    float bias_scale,
    float output_scale,
    int32_t output_zero_point,
    Tensor& out) {
  TIME_DECL(conv2d);
  TIME_START(conv2d);

  bool conv1d = input.dim() == 3;
  // input = [n, c, h, w]
  const int n = input.size(0);
  const int c = input.size(1);
  const int h = conv1d ? 1 : input.size(2);
  const int w = conv1d ? input.size(2) : input.size(3);
  // weight = [oc, wc, wh, ww]
  const int oc = weight.size(0);
  const int wc = weight.size(1);
  const int wh = conv1d ? 1 : weight.size(2);
  const int ww = conv1d ? weight.size(2) : weight.size(3);
  // output = [n, oc, oh, ow]
  const int oh = conv1d ? 1 : out.size(2);
  const int ow = conv1d ? out.size(2) : out.size(3);

#define typed_quantized_conv2d_nchw(ctype, dtype)                 \
  case ScalarType::dtype: {                                       \
    conv2d_nchw_core_generic<ctype, ctype, int32_t, ctype, true>( \
        input.const_data_ptr<ctype>(),                            \
        weight.const_data_ptr<ctype>(),                           \
        bias.const_data_ptr<int32_t>(),                           \
        out.mutable_data_ptr<ctype>(),                            \
        n,                                                        \
        c,                                                        \
        h,                                                        \
        w,                                                        \
        oc,                                                       \
        wc,                                                       \
        wh,                                                       \
        ww,                                                       \
        oh,                                                       \
        ow,                                                       \
        stride[0],                                                \
        stride[1],                                                \
        padding[0],                                               \
        padding[1],                                               \
        dilation[0],                                              \
        dilation[1],                                              \
        groups,                                                   \
        in_zero_point,                                            \
        weight_zero_point,                                        \
        bias_scale,                                               \
        output_scale,                                             \
        (ctype)output_zero_point);                                \
    break;                                                        \
  }
  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    case ScalarType::Char: {
#if CADENCE_CONV2D_GENERIC
      conv2d_nchw_core_generic<int8_t, int8_t, int32_t, int8_t, true>(
          input.const_data_ptr<int8_t>(),
          weight.const_data_ptr<int8_t>(),
          bias.const_data_ptr<int32_t>(),
          out.mutable_data_ptr<int8_t>(),
          n, c, h, w,
          oc, wc, wh, ww,
          oh, ow,
          stride[0], stride[1],
          padding[0], padding[1],
          dilation[0], dilation[1],
          groups,
          in_zero_point,
          weight_zero_point,
          bias_scale,
          output_scale,
          (int8_t)output_zero_point);
      break;
#endif

      const conv_layer_config_t* config_const = get_layer_config_by_params(
          c, h, w,              // ic, ih, iw
          oc, wh, ww,           // oc, kh, kw
          oh, ow,               // oh, ow
          stride[0], stride[1], // sy, sx
          padding[0], dilation[0]); // pad, dil

      // Make a mutable local copy — the static const table may reside in
      // read-only memory (.rodata), so writing through const_cast is undefined
      // behavior and silently fails on Xtensa targets.
      conv_layer_config_t config_local;
      conv_layer_config_t* config = NULL;
      float effective_scale = 0.0f;
      if (config_const != NULL) {
        config_local = *config_const;  // shallow copy all fields
        config = &config_local;

        // DMA path for all layers ≥ 4×4 spatial; generic C fallback for ≤ 2×2.
        //
        // XAI kernel pipeline: out = (acc >> accumShift) * outputScale >> outputShift
        // The kernel saturates the shifted accumulator to int16 [-32768, 32767]
        // after accumShift, so accumShift must be chosen to keep accumulators in range.
        effective_scale = bias_scale / output_scale;
      }

      if(config != NULL) {
        config->input_zero_point = static_cast<int>(in_zero_point);

        // Disable in-kernel ReLU — ExecuTorch applies ReLU as a separate op.
        config->relu_min = -128;
        config->relu_max = 127;

        // Bias correction: absorb input_zero_point and output_zero_point
        // into the kernel bias to avoid the double-clamp problem.
        // Also clamp to 24-bit range (ACC_INIT_BIAS takes lower 24 bits);
        // any residual beyond 24-bit is applied as post-kernel correction.
        const int32_t* bias_orig = bias.const_data_ptr<int32_t>();
        const int8_t*  wt_data   = weight.const_data_ptr<int8_t>();
        const int       wt_per_oc = weight.numel() / oc;

        static const int32_t BIAS_24BIT_MAX =  8388607;   // (1 << 23) - 1
        static const int32_t BIAS_24BIT_MIN = -8388608;   // -(1 << 23)

        // output_zero_point expressed in accumulator domain
        int64_t zp_acc_corr = 0;
        if (output_zero_point != 0 && effective_scale > 0.0f) {
          double zp_d = static_cast<double>(output_zero_point) / effective_scale;
          zp_acc_corr = static_cast<int64_t>(zp_d >= 0.0 ? zp_d + 0.5 : zp_d - 0.5);
        }

        // Per-channel split bias: kernel_bias (24-bit safe) + post_correction
        int32_t kernel_bias[2048];
        int32_t post_correction[2048];
        int64_t max_abs_kernel_bias = 0;
        for (int o = 0; o < oc; o++) {
          int32_t w_sum = 0;
          const int8_t* wt_oc = wt_data + o * wt_per_oc;
          for (int i = 0; i < wt_per_oc; i++) {
            w_sum += wt_oc[i];
          }
          int64_t bias_corr_64 = static_cast<int64_t>(bias_orig[o])
                               - static_cast<int64_t>(in_zero_point) * w_sum;

          int64_t target_bias = bias_corr_64 + zp_acc_corr;

          int32_t kb;
          if (target_bias > BIAS_24BIT_MAX) {
            kb = BIAS_24BIT_MAX;
          } else if (target_bias < BIAS_24BIT_MIN) {
            kb = BIAS_24BIT_MIN;
          } else {
            kb = static_cast<int32_t>(target_bias);
          }
          kernel_bias[o] = kb;

          int64_t abs_kb = kb >= 0 ? kb : -static_cast<int64_t>(kb);
          if (abs_kb > max_abs_kernel_bias) max_abs_kernel_bias = abs_kb;

          int64_t bias_residual = target_bias - kb;
          float resid_float = static_cast<float>(bias_residual) * effective_scale;
          int32_t resid_int = static_cast<int32_t>(resid_float >= 0.0f
                            ? resid_float + 0.5f : resid_float - 0.5f);
          post_correction[o] = resid_int;
        }

        // accumShift: ensure (acc >> accSh) fits in int16 after PACK.
        // Tight bound from actual weight L1 norms instead of worst-case 128*128*P.
        // max_acc = |bias| + sum(|weight_i|) * 128 since inputs are int8 (magnitude ≤ 128).
        // Compute max sum(|weights|) across all output channels
        int64_t max_weight_l1 = 0;
        for (int o = 0; o < oc; o++) {
          const int8_t* wt_oc = wt_data + o * wt_per_oc;
          int64_t w_l1 = 0;
          for (int i = 0; i < wt_per_oc; i++) {
            w_l1 += (wt_oc[i] >= 0) ? wt_oc[i] : -wt_oc[i];
          }
          if (w_l1 > max_weight_l1) max_weight_l1 = w_l1;
        }

        // Tight max accumulator bound: bias + L1(weights) * max_input_magnitude
        float max_acc = static_cast<float>(max_abs_kernel_bias)
                      + static_cast<float>(max_weight_l1) * 128.0f;

        int accum_shift = 0;
        while (max_acc / static_cast<float>(1LL << accum_shift) > 32767.0f
               && accum_shift < 31) {
          accum_shift++;
        }

        config->accum_shift = accum_shift;

        // outputShift & outputScale: maximize precision within uint16 range.
        int best_shift = 15;
        int64_t total_shift = static_cast<int64_t>(accum_shift) + best_shift;
        int32_t raw_scale = static_cast<int32_t>(
            effective_scale * static_cast<double>(1LL << total_shift));
        if (raw_scale > 65535) {
          // Scale too large for uint16_t, reduce outputShift until it fits
          while (best_shift > 0 && raw_scale > 65535) {
            best_shift--;
            total_shift = static_cast<int64_t>(accum_shift) + best_shift;
            raw_scale = static_cast<int32_t>(
                effective_scale * static_cast<double>(1LL << total_shift));
          }
        } else if (raw_scale < 16384 && best_shift < 31) {
          // Scale too small, increase outputShift for better precision
          while (best_shift < 31) {
            int64_t trial_total = static_cast<int64_t>(accum_shift) + best_shift + 1;
            if (trial_total > 62) break;  // avoid 1LL << overflow
            int32_t trial = static_cast<int32_t>(
                effective_scale * static_cast<double>(1LL << trial_total));
            if (trial > 65535) break;
            best_shift++;
            raw_scale = trial;
          }
        }
        if (raw_scale <= 0) raw_scale = 1;
        if (raw_scale > 65535) raw_scale = 65535;

        config->output_shift = best_shift;
        config->output_scale = raw_scale;

        // CPU-computed kernel_bias resides only in cache;
        // DMA bypasses cache and reads system memory, so writeback is needed.
        xthal_dcache_region_writeback(
            reinterpret_cast<int8_t*>(kernel_bias),
            oc * sizeof(int32_t));

        XAI_ERR_TYPE kern_status = conv_execute_kernel(
            const_cast<int8_t*>(input.const_data_ptr<int8_t>()),
            out.mutable_data_ptr<int8_t>(),
            const_cast<int8_t*>(weight.const_data_ptr<int8_t>()),
            reinterpret_cast<int8_t*>(kernel_bias),
            config);
        if (kern_status != 0) {
          printf("*** conv_execute_kernel FAILED for %s: status=%d ***\n",
                 config->layer_name ? config->layer_name : "?", (int)kern_status);
        }

        // Invalidate cache for DMA-written output so post-correction
        // and next operator see fresh data instead of stale cache lines
        xthal_dcache_region_invalidate(
            out.mutable_data_ptr<int8_t>(),
            n * oc * oh * ow * sizeof(int8_t));

        // Apply post-kernel residual correction
        for (int _n = 0; _n < n; _n++) {
          for (int _oc = 0; _oc < oc; _oc++) {
            int32_t corr = post_correction[_oc];
            if (corr == 0) continue;
            int8_t* ch_out = out.mutable_data_ptr<int8_t>() + (_n * oc * oh * ow + _oc * oh * ow);
            for (int _s = 0; _s < oh * ow; _s++) {
              int32_t val = static_cast<int32_t>(ch_out[_s]) + corr;
              val = val < -128 ? -128 : (val > 127 ? 127 : val);
              ch_out[_s] = static_cast<int8_t>(val);
            }
          }
        }

        break;
      }
      // Fall through to generic implementation
      conv2d_nchw_core_generic<int8_t, int8_t, int32_t, int8_t, true>(
          input.const_data_ptr<int8_t>(),
          weight.const_data_ptr<int8_t>(),
          bias.const_data_ptr<int32_t>(),
          out.mutable_data_ptr<int8_t>(),
          n, c, h, w,
          oc, wc, wh, ww,
          oh, ow,
          stride[0], stride[1],
          padding[0], padding[1],
          dilation[0], dilation[1],
          groups,
          in_zero_point,
          weight_zero_point,
          bias_scale,
          output_scale,
          (int8_t)output_zero_point);
      break;
    }
    // Handle uint8_t (Byte) case - previously covered by ET_FORALL_CADENCE_QUANTIZED_TYPES
    // Note: Char (int8_t) is handled explicitly above with optimized kernel
    typed_quantized_conv2d_nchw(uint8_t, Byte);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_conv2d_nchw

  TIME_END(conv2d);
  TIME_DISPLAY(conv2d, input.numel(), "elements");
}

void quantized_conv2d_nchw_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED const Tensor& out_multiplier,
    __ET_UNUSED const Tensor& out_shift,
    Tensor& out) {
  const float bias_scale_float = bias_scale.const_data_ptr<float>()[0];
  const int32_t weight_zero_point_int =
      weight_zero_point.const_data_ptr<int32_t>()[0];
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point_int,
      bias_scale_float,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

void quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  quantized_conv2d_nchw(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out);
}

} // namespace native
} // namespace vision

namespace generic {
namespace native {

void quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  ::impl::vision::native::quantized_conv2d_nchw(
      input, weight, bias, stride, padding, dilation, groups,
      in_zero_point, weight_zero_point, bias_scale, output_scale,
      output_zero_point, out);
}

void quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  ::impl::vision::native::quantized_conv2d_nchw(
      input, weight, bias, stride, padding, dilation, groups,
      in_zero_point, weight_zero_point, bias_scale, output_scale,
      output_zero_point, out);
}

void quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  ::impl::vision::native::quantized_conv2d_nchw(
      input, weight, bias, stride, padding, dilation, groups,
      in_zero_point, weight_zero_point, bias_scale, output_scale,
      output_zero_point, out);
}

void quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  ::impl::vision::native::quantized_conv2d_nchw(
      input, weight, bias, stride, padding, dilation, groups,
      in_zero_point, weight_zero_point, bias_scale, output_scale,
      output_zero_point, out);
}

} // namespace native
} // namespace generic
} // namespace impl
