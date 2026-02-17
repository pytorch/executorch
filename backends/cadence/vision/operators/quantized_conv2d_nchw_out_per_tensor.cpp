/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <lib.h>
#include <stdio.h>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/operators.h>
#include <executorch/backends/cadence/vision/operators/conv/conv_layer_configs.h>

// CSV logging for per-layer input/output data
static int csv_layer_counter = 0;
static FILE* csv_log_file = NULL;

static void csv_log_open() {
  if (csv_log_file == NULL) {
    csv_log_file = fopen("conv2d_layer_dump.csv", "w");
    if (csv_log_file) {
      fprintf(csv_log_file, "layer,direction,n,c,h,w,oc,wh,ww,oh,ow,groups,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,in_zp,wt_zp,bias_scale,out_scale,out_zp,numel,data\n");
    }
  }
}

static void csv_log_tensor_row(
    int layer, const char* direction,
    int n, int c, int h, int w, int oc, int wh, int ww, int oh, int ow,
    int groups, int sh, int sw, int ph, int pw, int dh, int dw,
    int in_zp, int wt_zp, float bias_sc, float out_sc, int out_zp,
    const int8_t* data, int numel) {
  csv_log_open();
  if (!csv_log_file) return;
  fprintf(csv_log_file, "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%d,%d,",
      layer, direction, n, c, h, w, oc, wh, ww, oh, ow,
      groups, sh, sw, ph, pw, dh, dw, in_zp, wt_zp, bias_sc, out_sc, out_zp, numel);
  for (int i = 0; i < numel; i++) {
    if (i > 0) fprintf(csv_log_file, " ");
    fprintf(csv_log_file, "%d", (int)data[i]);
  }
  fprintf(csv_log_file, "\n");
  fflush(csv_log_file);
}

static void csv_log_close() {
  if (csv_log_file) {
    fclose(csv_log_file);
    csv_log_file = NULL;
  }
}


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

  printf("quantized_conv2d_nchw: n=%d, c=%d, h=%d, w=%d, oc=%d, wc=%d, wh=%d, ww=%d, oh=%d, ow=%d\n",
         n, c, h, w, oc, wc, wh, ww, oh, ow);
  printf("quantized_conv2d_nchw: groups=%d, in_zero_point=%d, weight_zero_point=%d, bias_scale=%f, output_scale=%f\n",
         groups, in_zero_point, weight_zero_point, bias_scale, output_scale);

  // Log input tensor to CSV
  int cur_layer = csv_layer_counter++;
  if (input.scalar_type() == ScalarType::Char) {
    csv_log_tensor_row(cur_layer, "input",
        n, c, h, w, oc, wh, ww, oh, ow,
        groups, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_zero_point, weight_zero_point, bias_scale, output_scale, output_zero_point,
        input.const_data_ptr<int8_t>(), input.numel());
  }

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
  printf("quantized_conv2d_nchw: output dtype=%d\n", static_cast<int>(dtype));
  switch (dtype) {
    case ScalarType::Char: {
#if CADENCE_CONV2D_GENERIC
      printf("quantized_conv2d_nchw: Using generic conv2d implementation (CADENCE_CONV2D_GENERIC enabled)\n");
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
      printf("quantized_conv2d_nchw: 1234 trying to use optimized cadence kernel for QInt8\n");
      // if(wh == 1 && ww == 1) {

      //   const conv_layer_config_t* config get_layer_config(n,c,h,w,oc,wc,wh,ww,oh,ow,stride,padding,dilation);
      //   // For 1x1 conv, use int8_t to match QNNPACK behavior
      //   //function call

      //   break;
      // } else if( wh == 3 && ww ==3) {
      //   // For 3x3 conv, use int8_t to match QNNPACK behavior
      //   //function call

      //   break;
      // } else if( wh == 7 && ww ==7) {
      //   // For other kernel sizes, use int32_t to avoid overflow
      //   //function call

      //   break;
      // }else {
      //   // Default to int32_t
      //   // Do nothing here, fall through to general case
      // }

        // // By parameters (conv1: 3ch 224x224 → 64ch 112x112, 7x7 kernel, stride 2, pad 3)
        // const conv_layer_config_t* config = get_layer_config_by_params(
        // 3, 224, 224,    // ic, ih, iw
        // 64, 7, 7,       // oc, kh, kw
        // 112, 112,       // oh, ow
        // 2, 2,           // sy, sx
        // 3, 1            // pad, dil
        // );

      
      printf("quantized_conv2d_nchw: searching for optimized cadence kernel for QInt8\n");
      // print first 100 elements of input tensor for debugging
      printf("Input tensor first 100 elements:\n");
      const int input_numel = input.numel();
      const int8_t* input_data = input.const_data_ptr<int8_t>();
      for (int i = 0; i < std::min(100, input_numel); i++) {
          printf("%d ", input_data[i]);
      }
      printf("\n");

      //print weight anf bias tensor first 100 elements for debugging
      printf("Weight tensor first 100 elements:\n");
      const int weight_numel = weight.numel();
      const int8_t* weight_data = weight.const_data_ptr<int8_t>();
      for (int i = 0; i < std::min(100, weight_numel); i++) {
          printf("%d ", weight_data[i]);
      }
      printf("\n");

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
      if (config_const != NULL) {
        config_local = *config_const;  // shallow copy all fields
        config = &config_local;

        // Compute fixed-point effective scale = bias_scale / output_scale
        // XAI kernel pipeline: out = (acc >> accumShift) * outputScale >> outputShift
        //
        // IMPORTANT: The XAI kernel internally saturates the shifted accumulator
        // to int16 range [-32768, 32767] after accumShift.  With accumShift=0,
        // layers with many input channels (ic * kh * kw large) produce int32
        // accumulators that exceed int16, causing output clipping.
        //
        // We must choose accumShift so that the worst-case accumulator fits in
        // int16 after shifting.  The effective_scale relationship is:
        //   effective_scale = outputScale / 2^(accumShift + outputShift)
        float effective_scale = bias_scale / output_scale;

        // --- Step 1: Compute accumShift to keep shifted acc within int16 ---
        // Worst-case accumulator magnitude for int8 × int8 MAC + bias correction:
        //   max_acc = num_products * 127 * (127 + |in_zero_point|)
        // The bias correction adds |-zp * sum(weights)| ≤ |zp| * num_products * 127
        // to the accumulator, so both MAC and correction terms must fit after shift.
        int32_t icpg = c / groups;          // input channels per group
        int32_t num_products = icpg * wh * ww;
        int32_t abs_zp = (in_zero_point >= 0)
                       ? static_cast<int32_t>(in_zero_point)
                       : static_cast<int32_t>(-in_zero_point);
        float max_acc = static_cast<float>(num_products) * 127.0f * (127.0f + static_cast<float>(abs_zp));

        int accum_shift = 0;
        while (max_acc / static_cast<float>(1LL << accum_shift) > 32767.0f
               && accum_shift < 31) {
          accum_shift++;
        }
        config->accum_shift = accum_shift;

        // --- Step 2: Find outputShift & outputScale for best precision ---
        // outputScale = effective_scale * 2^(accumShift + outputShift)
        // outputScale must fit in uint16_t [1, 65535].
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
        // Clamp to valid uint16_t range [1, 65535]
        if (raw_scale <= 0) raw_scale = 1;
        if (raw_scale > 65535) raw_scale = 65535;

        config->output_shift = best_shift;
        config->output_scale = raw_scale;
        printf("quantized_conv2d_nchw: effective_scale=%f, output_scale=%d, output_shift=%d, accum_shift=%d (ic=%d kh=%d kw=%d num_products=%d)\n",
               effective_scale, config->output_scale, config->output_shift, config->accum_shift,
               (int)icpg, (int)wh, (int)ww, (int)num_products);
      }

      printf("quantized_conv2d_nchw: obtained layer config %p\n", (void*)config);

      //ptint config feilds for debugging
      printf("Layer ID: %d\n", config->layer_id);
      printf("Layer Name: %s\n", config->layer_name);
      printf("Kernel Name: %s\n", config->kernel_name);
      printf("Config Key: %s\n", config->config_key);
      printf("Input Dimensions (DRAM): %d x %d x %d\n",
              config->src_dim3_size, config->src_dim2_size, config->src_dim1_size); 
      printf("Output Dimensions (DRAM): %d x %d x %d\n",
              config->dst_dim3_size, config->dst_dim2_size, config->dst_dim1_size);
    
      if(config != NULL) {
        // Set input_zero_point on config so kernel executors can fill padding
        // with the correct value (the quantized representation of 0.0).
        config->input_zero_point = static_cast<int>(in_zero_point);

        // --- Bias correction for zero-point handling ---
        // Instead of subtracting in_zero_point from int8 input (which overflows
        // for zero_point=-128 since the result range [0,255] exceeds int8),
        // we mathematically absorb the zero-point into the bias:
        //   acc = Σ((input - zp) * weight) + bias
        //       = Σ(input * weight) - zp * Σ(weight) + bias
        //       = Σ(input * weight) + (bias - zp * Σ(weight))
        // The kernel then uses raw (uncorrected) input data with corrected bias.
        // Padding is filled with in_zero_point so padded positions represent 0.0.
        const int32_t* bias_orig = bias.const_data_ptr<int32_t>();
        const int8_t*  wt_data   = weight.const_data_ptr<int8_t>();
        const int       wt_per_oc = weight_numel / oc;
        int32_t bias_corrected[2048]; // Stack buffer, max 2048 output channels

        for (int o = 0; o < oc; o++) {
          int32_t w_sum = 0;
          const int8_t* wt_oc = wt_data + o * wt_per_oc;
          for (int i = 0; i < wt_per_oc; i++) {
            w_sum += wt_oc[i];
          }
          bias_corrected[o] = bias_orig[o]
                            - static_cast<int32_t>(in_zero_point) * w_sum;
        }
        printf("quantized_conv2d_nchw: bias correction applied (in_zero_point=%d, weight_zero_point=%d)\n",
               (int)in_zero_point, (int)weight_zero_point);

        // Use optimized cadence kernel dma/cache
        // Pass raw input (no zero-point subtraction) and corrected bias.
        conv_execute_kernel(
            const_cast<int8_t*>(input.const_data_ptr<int8_t>()),
            out.mutable_data_ptr<int8_t>(),
            const_cast<int8_t*>(weight.const_data_ptr<int8_t>()),
            reinterpret_cast<int8_t*>(bias_corrected),
            config);

        // XAI kernel operates in symmetric mode (no output offset).
        // Add output_zero_point to convert back to asymmetric quantization.
        if (output_zero_point != 0) {
          int8_t* out_data = out.mutable_data_ptr<int8_t>();
          const int out_numel = out.numel();
          for (int i = 0; i < out_numel; i++) {
            int32_t val = static_cast<int32_t>(out_data[i]) + output_zero_point;
            // Clamp to int8 range [-128, 127]
            val = val < -128 ? -128 : (val > 127 ? 127 : val);
            out_data[i] = static_cast<int8_t>(val);
          }
        }

        printf("Using optimized cadence conv2d kernel for Char (output_zero_point=%d applied)\n",
               (int)output_zero_point);
        break;
      }
      printf("No optimized cadence conv2d kernel found for Char, using generic implementation\n");
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
      printf("quantized_conv2d_nchw: unsupported dtype %d\n", static_cast<int>(dtype));
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_conv2d_nchw

  // Log output tensor to CSV
  if (out.scalar_type() == ScalarType::Char) {
    csv_log_tensor_row(cur_layer, "output",
        n, c, h, w, oc, wh, ww, oh, ow,
        groups, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_zero_point, weight_zero_point, bias_scale, output_scale, output_zero_point,
        out.const_data_ptr<int8_t>(), out.numel());
  }
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
} // namespace impl
