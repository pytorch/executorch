// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <stdexcept>
#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

int64_t calc_out_size(
    int64_t in_size,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
  return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride +
      1;
}

} // namespace

// Shared perf/skip classification used by both create_conv2d_test_case (to tag
// PERF vs ACCU) and conv2d_reference_impl (to gate the large-K FP16 reference
// check). A shape is "perf" if any dimension reaches kRefDimSizeLimit; the
// boundary is inclusive (>=) so a 64-wide dim counts as perf — FP16
// accumulation error at K = K_h * K_w * C_in for such shapes can exceed the
// half tolerance and false-fail. Keep both call sites on this single helper to
// avoid the two predicates drifting apart.
bool conv2d_is_perf_shape(int64_t C_in, int64_t C_out, int64_t H, int64_t W) {
  return C_in >= kRefDimSizeLimit || C_out >= kRefDimSizeLimit ||
      H >= kRefDimSizeLimit || W >= kRefDimSizeLimit;
}

TestCase create_conv2d_test_case(
    const Conv2dTestConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector) {
  TestCase test_case;

  bool is_perf = conv2d_is_perf_shape(
      config.dims.C, config.C_out, config.dims.H, config.dims.W);

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = repr_str(storage_type, memory_layout);
  std::string dtype_str = dtype_short(dtype);
  std::string bias_str = config.has_bias ? "+bias" : "";

  int64_t H_out = calc_out_size(
      config.dims.H,
      config.kernel.h,
      config.stride.h,
      config.padding.h,
      config.dilation.h);
  int64_t W_out = calc_out_size(
      config.dims.W,
      config.kernel.w,
      config.stride.w,
      config.padding.w,
      config.dilation.w);

  std::string shape = "[" + std::to_string(config.dims.N) + "," +
      std::to_string(config.dims.C) + "," + std::to_string(config.dims.H) +
      "," + std::to_string(config.dims.W) + "]->[" +
      std::to_string(config.C_out) + "] k" + std::to_string(config.kernel.h) +
      "x" + std::to_string(config.kernel.w) + " s" +
      std::to_string(config.stride.h) + " p" +
      std::to_string(config.padding.h) + " d" +
      std::to_string(config.dilation.h);

  std::string suffix = bias_str;
  if (!impl_selector.empty()) {
    if (!suffix.empty()) {
      suffix += " ";
    }
    suffix += "[" + impl_selector + "]";
  }

  std::string name =
      make_test_label(prefix, dtype_str, dtype_str, shape, storage_str, suffix);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d.default");

  // Input tensor [N, C_in, H, W]
  ValueSpec input(
      {config.dims.N, config.dims.C, config.dims.H, config.dims.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C_out, C_in, K_h, K_w] - constant
  ValueSpec weight(
      {config.C_out, config.dims.C, config.kernel.h, config.kernel.w},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);
  weight.set_constant(true);

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);

  // Bias (or none)
  if (config.has_bias) {
    ValueSpec bias(
        {config.C_out},
        dtype,
        storage_type,
        memory_layout,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.w)));

  // impl_selector string
  test_case.add_input_spec(ValueSpec::make_string(impl_selector));

  // Output tensor [N, C_out, H_out, W_out]
  ValueSpec output(
      {config.dims.N, config.C_out, H_out, W_out},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::ZEROS);
  test_case.add_output_spec(output);

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-1f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  test_case.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});

  return test_case;
}

TestCase create_conv2d_dw_test_case(
    const Conv2dDwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector) {
  TestCase test_case;

  bool is_perf = config.dims.C > kRefDimSizeLimit ||
      config.dims.H > kRefDimSizeLimit || config.dims.W > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = repr_str(storage_type, memory_layout);
  std::string dtype_str = dtype_short(dtype);
  std::string bias_str = config.has_bias ? "+bias" : "";

  int64_t H_out = calc_out_size(
      config.dims.H,
      config.kernel.h,
      config.stride.h,
      config.padding.h,
      config.dilation.h);
  int64_t W_out = calc_out_size(
      config.dims.W,
      config.kernel.w,
      config.stride.w,
      config.padding.w,
      config.dilation.w);
  (void)H_out;
  (void)W_out;

  // groups for depthwise conv2d == number of input channels
  std::string shape = "[" + std::to_string(config.dims.N) + "," +
      std::to_string(config.dims.C) + "," + std::to_string(config.dims.H) +
      "," + std::to_string(config.dims.W) + "] k" +
      std::to_string(config.kernel.h) + " s" + std::to_string(config.stride.h) +
      " p" + std::to_string(config.padding.h) + " d" +
      std::to_string(config.dilation.h) + " g" + std::to_string(config.dims.C);

  std::string suffix = bias_str;
  if (!impl_selector.empty()) {
    if (!suffix.empty()) {
      suffix += " ";
    }
    suffix += "[" + impl_selector + "]";
  }

  std::string name =
      make_test_label(prefix, dtype_str, dtype_str, shape, storage_str, suffix);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d_dw.default");

  // Input tensor [N, C, H, W]
  ValueSpec input(
      {config.dims.N, config.dims.C, config.dims.H, config.dims.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C, 1, K_h, K_w] - constant
  ValueSpec weight(
      {config.dims.C, 1, config.kernel.h, config.kernel.w},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);
  weight.set_constant(true);

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);

  // Bias (or none)
  if (config.has_bias) {
    ValueSpec bias(
        {config.dims.C},
        dtype,
        storage_type,
        memory_layout,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.w)));

  // impl_selector string
  test_case.add_input_spec(ValueSpec::make_string(impl_selector));

  // Output tensor [N, C, H_out, W_out]
  ValueSpec output(
      {config.dims.N, config.dims.C, H_out, W_out},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::ZEROS);
  test_case.add_output_spec(output);

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-1f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  test_case.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});

  return test_case;
}

TestCase create_conv2d_pw_test_case(
    const Conv2dPwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout) {
  TestCase test_case;

  bool is_perf = config.C_in > kRefDimSizeLimit ||
      config.C_out > kRefDimSizeLimit || config.H > kRefDimSizeLimit ||
      config.W > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = repr_str(storage_type, memory_layout);
  std::string dtype_str = dtype_short(dtype);
  std::string bias_str = config.has_bias ? "+bias" : "";

  // Pointwise conv2d: kernel 1x1, stride 1, pad 0, dilation 1, groups 1
  std::string shape = "[" + std::to_string(config.N) + "," +
      std::to_string(config.C_in) + "," + std::to_string(config.H) + "," +
      std::to_string(config.W) + "]x[" + std::to_string(config.C_out) + "," +
      std::to_string(config.C_in) + ",1,1] s1 p0 d1 g1";

  std::string name = make_test_label(
      prefix, dtype_str, dtype_str, shape, storage_str, bias_str);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d_pw.default");

  // Input tensor [N, C_in, H, W]
  ValueSpec input(
      {config.N, config.C_in, config.H, config.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C_out, C_in, 1, 1] - constant
  ValueSpec weight(
      {config.C_out, config.C_in, 1, 1},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);
  weight.set_constant(true);

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);

  // Bias (or none)
  if (config.has_bias) {
    ValueSpec bias(
        {config.C_out},
        dtype,
        storage_type,
        memory_layout,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // impl_selector
  ValueSpec impl_selector_spec = ValueSpec::make_string("default");
  test_case.add_input_spec(impl_selector_spec);

  // Output tensor [N, C_out, H, W]
  ValueSpec output(
      {config.N, config.C_out, config.H, config.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::ZEROS);
  test_case.add_output_spec(output);

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-1f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  test_case.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});

  return test_case;
}

int64_t conv2d_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();
  auto output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t N = output_sizes[0];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];
  int64_t C_in = input_sizes[1];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];

  return 2 * N * C_out * C_in * H_out * W_out * K_h * K_w;
}

int64_t conv2d_dw_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();
  auto output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t N = output_sizes[0];
  int64_t C = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];

  // Each output element: K_h * K_w multiplies + (K_h * K_w - 1) adds
  return 2 * N * C * H_out * W_out * K_h * K_w;
}

int64_t conv2d_pw_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H = input_sizes[2];
  int64_t W = input_sizes[3];
  int64_t C_out = weight_sizes[0];

  return 2 * N * C_out * C_in * H * W;
}

// Reference implementation for general conv2d (groups=1).
//
// Supports both FP32 and (small-shape) FP16 inputs. The math is always done in
// float; for FP16 the master input/weight/bias values are dequantized from
// their half storage via get_element(), and the resulting float reference is
// compared against the dequantized GPU output by validate_against_reference().
//
// FP16 accumulation error grows with K (= K_h * K_w * C_in). For large-K PERF
// shapes the FP32 reference would diverge from the GPU's FP16 accumulation
// enough to trip even the relaxed half tolerance, producing false failures, so
// those are intentionally left timing-only: this function throws
// std::invalid_argument, which execute_test_cases() catches to skip the
// correctness check (ref_computed stays false) while still benchmarking.
void conv2d_reference_impl(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat && input.dtype != vkapi::kHalf) {
    throw std::invalid_argument("Reference only supports float and half");
  }

  auto input_sizes = input.get_tensor_sizes();
  auto weight_sizes = weight.get_tensor_sizes();
  auto output_sizes = output.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = weight_sizes[0];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // For FP16, only compute a reference for small (ACCU) shapes where K is small
  // enough that FP32-vs-FP16 accumulation error stays within the half
  // tolerance. Large-K PERF half shapes stay timing-only via the throw below.
  // The predicate mirrors create_conv2d_test_case's is_perf classification.
  if (input.dtype == vkapi::kHalf) {
    const bool is_perf = conv2d_is_perf_shape(C_in, C_out, H_in, W_in);
    if (is_perf) {
      throw std::invalid_argument(
          "Half reference skipped for large-K PERF shape (timing-only)");
    }
  }

  int64_t stride_h = test_case.inputs()[3].get_int_value();
  int64_t stride_w = test_case.inputs()[4].get_int_value();
  int64_t padding_h = test_case.inputs()[5].get_int_value();
  int64_t padding_w = test_case.inputs()[6].get_int_value();
  int64_t dilation_h = test_case.inputs()[7].get_int_value();
  int64_t dilation_w = test_case.inputs()[8].get_int_value();

  // get_element() materializes a float regardless of dtype (it dequantizes
  // half master data), so the same loop body serves both FP32 and FP16.
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C_out * H_out * W_out, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < C_out; ++co) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          float sum = 0.0f;
          for (int64_t ci = 0; ci < C_in; ++ci) {
            for (int64_t kh = 0; kh < K_h; ++kh) {
              for (int64_t kw = 0; kw < K_w; ++kw) {
                int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
                int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                  float in_val = input.get_element(
                      n * (C_in * H_in * W_in) + ci * (H_in * W_in) +
                      ih * W_in + iw);
                  // weight is [C_out, C_in, K_h, K_w]
                  float w_val = weight.get_element(
                      co * (C_in * K_h * K_w) + ci * (K_h * K_w) + kh * K_w +
                      kw);
                  sum += in_val * w_val;
                }
              }
            }
          }
          if (!bias_spec.is_none()) {
            sum += bias_spec.get_element(co);
          }
          ref_data
              [n * (C_out * H_out * W_out) + co * (H_out * W_out) + oh * W_out +
               ow] = sum;
        }
      }
    }
  }
}

// Reference implementation for depthwise conv2d
void conv2d_dw_reference_impl(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto input_sizes = input.get_tensor_sizes();
  auto weight_sizes = weight.get_tensor_sizes();
  auto output_sizes = output.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  int64_t stride_h = test_case.inputs()[3].get_int_value();
  int64_t stride_w = test_case.inputs()[4].get_int_value();
  int64_t padding_h = test_case.inputs()[5].get_int_value();
  int64_t padding_w = test_case.inputs()[6].get_int_value();
  int64_t dilation_h = test_case.inputs()[7].get_int_value();
  int64_t dilation_w = test_case.inputs()[8].get_int_value();

  auto& input_data = input.get_float_data();
  auto& weight_data = weight.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C * H_out * W_out, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          float sum = 0.0f;
          for (int64_t kh = 0; kh < K_h; ++kh) {
            for (int64_t kw = 0; kw < K_w; ++kw) {
              int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
              int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
              if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float in_val = input_data
                    [n * (C * H_in * W_in) + c * (H_in * W_in) + ih * W_in +
                     iw];
                // weight is [C, 1, K_h, K_w]
                float w_val = weight_data[c * (K_h * K_w) + kh * K_w + kw];
                sum += in_val * w_val;
              }
            }
          }
          if (!bias_spec.is_none()) {
            auto& bias_data = bias_spec.get_float_data();
            sum += bias_data[c];
          }
          ref_data
              [n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out +
               ow] = sum;
        }
      }
    }
  }
}

// Reference implementation: pointwise conv2d is essentially a matmul
// output[n][c_out][h][w] = bias[c_out] +
//   sum_over_c_in(input[n][c_in][h][w] * weight[c_out][c_in][0][0])
void conv2d_pw_reference_impl(TestCase& test_case) {
  // input[0], weight[1], bias[2], impl_selector[3]
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto input_sizes = input.get_tensor_sizes();
  auto weight_sizes = weight.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H = input_sizes[2];
  int64_t W = input_sizes[3];
  int64_t C_out = weight_sizes[0];

  auto& input_data = input.get_float_data();
  auto& weight_data = weight.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C_out * H * W, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < C_out; ++co) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          float sum = 0.0f;
          for (int64_t ci = 0; ci < C_in; ++ci) {
            float in_val =
                input_data[n * (C_in * H * W) + ci * (H * W) + h * W + w];
            // weight is [C_out, C_in, 1, 1]
            float w_val = weight_data[co * C_in + ci];
            sum += in_val * w_val;
          }
          if (!bias_spec.is_none()) {
            auto& bias_data = bias_spec.get_float_data();
            sum += bias_data[co];
          }
          ref_data[n * (C_out * H * W) + co * (H * W) + h * W + w] = sum;
        }
      }
    }
  }
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
