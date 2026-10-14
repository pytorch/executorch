// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv1d/conv1d.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv1d {

TestCase create_conv1d_dw_test_case(
    const Conv1dDWConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type) {
  TestCase test_case;

  bool is_perf = config.C > kRefDimSizeLimit || config.L > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type) + "(HP)";
  std::string dtype_str = dtype_short(dtype);
  std::string bias_str = config.has_bias ? "+bias" : "";

  int64_t L_out =
      (config.L + 2 * config.padding - config.dilation * (config.K - 1) - 1) /
          config.stride +
      1;

  std::string shape = "[" + std::to_string(config.N) + "," +
      std::to_string(config.C) + "," + std::to_string(config.L) + "] k" +
      std::to_string(config.K) + " s" + std::to_string(config.stride) + " p" +
      std::to_string(config.padding) + " d" + std::to_string(config.dilation);

  std::string name = make_test_label(
      prefix, dtype_str, dtype_str, shape, storage_str, bias_str);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv1d_dw.default");

  // Input: [N, C, L] height-packed
  ValueSpec input(
      {config.N, config.C, config.L},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  test_case.add_input_spec(input);

  // Weight: [C, 1, K] height-packed, constant
  ValueSpec weight(
      {config.C, 1, config.K},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  weight.set_constant(true);
  test_case.add_input_spec(weight);

  // Bias: [C] or None
  if (config.has_bias) {
    ValueSpec bias(
        {config.C},
        dtype,
        storage_type,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride
  test_case.add_input_spec(
      ValueSpec(std::vector<int32_t>{static_cast<int32_t>(config.stride)}));
  // padding
  test_case.add_input_spec(
      ValueSpec(std::vector<int32_t>{static_cast<int32_t>(config.padding)}));
  // dilation
  test_case.add_input_spec(
      ValueSpec(std::vector<int32_t>{static_cast<int32_t>(config.dilation)}));
  // groups = C (depthwise)
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.C)));

  // Output: [N, C, L_out] height-packed
  ValueSpec output(
      {config.N, config.C, L_out},
      dtype,
      storage_type,
      utils::kHeightPacked,
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

TestCase create_conv1d_pw_test_case(
    const Conv1dPWConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type) {
  TestCase test_case;

  bool is_perf = config.C_in > kRefDimSizeLimit ||
      config.C_out > kRefDimSizeLimit || config.L > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type) + "(HP)";
  std::string dtype_str = dtype_short(dtype);

  std::string bias_str = config.has_bias ? "+bias" : "";

  std::string shape = "[" + std::to_string(config.N) + "," +
      std::to_string(config.C_in) + "," + std::to_string(config.L) + "]x[" +
      std::to_string(config.C_out) + "," + std::to_string(config.C_in) + ",1]";

  std::string name = make_test_label(
      prefix, dtype_str, dtype_str, shape, storage_str, bias_str);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv1d_pw.default");

  // Input: [N, C_in, L] height-packed
  ValueSpec input(
      {config.N, config.C_in, config.L},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  test_case.add_input_spec(input);

  // Weight: [C_out, C_in, 1] height-packed, constant
  ValueSpec weight(
      {config.C_out, config.C_in, 1},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  weight.set_constant(true);
  test_case.add_input_spec(weight);

  // Bias: [C_out] or None
  if (config.has_bias) {
    ValueSpec bias(
        {config.C_out},
        dtype,
        storage_type,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride = [1]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{1}));
  // padding = [0]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{0}));
  // dilation = [1]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{1}));
  // groups = 1
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(1)));

  // Output: [N, C_out, L] height-packed
  ValueSpec output(
      {config.N, config.C_out, config.L},
      dtype,
      storage_type,
      utils::kHeightPacked,
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

int64_t conv1d_dw_flop_calculator(const TestCase& test_case) {
  auto in_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto w_sizes = test_case.inputs()[1].get_tensor_sizes();
  auto out_sizes = test_case.outputs()[0].get_tensor_sizes();

  const int64_t N = in_sizes[0];
  const int64_t C = in_sizes[1];
  const int64_t K = w_sizes[2];
  const int64_t L_out = out_sizes[2];

  return 2 * N * C * L_out * K;
}

int64_t conv1d_pw_flop_calculator(const TestCase& test_case) {
  auto in_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto w_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t N = in_sizes[0];
  int64_t C_in = in_sizes[1];
  int64_t L = in_sizes[2];
  int64_t C_out = w_sizes[0];

  return 2 * N * C_in * C_out * L;
}

void conv1d_dw_reference_impl(TestCase& test_case) {
  const auto& input_spec = test_case.inputs()[0];
  const auto& weight_spec = test_case.inputs()[1];
  const auto& bias_spec = test_case.inputs()[2];
  const auto& stride_spec = test_case.inputs()[3];
  const auto& padding_spec = test_case.inputs()[4];
  const auto& dilation_spec = test_case.inputs()[5];
  ValueSpec& output = test_case.outputs()[0];

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto in_sizes = input_spec.get_tensor_sizes();
  auto w_sizes = weight_spec.get_tensor_sizes();
  auto out_sizes = output.get_tensor_sizes();

  const int64_t N = in_sizes[0];
  const int64_t C = in_sizes[1];
  const int64_t L_in = in_sizes[2];
  const int64_t K = w_sizes[2];
  const int64_t L_out = out_sizes[2];

  const int64_t stride = stride_spec.get_int_list()[0];
  const int64_t padding = padding_spec.get_int_list()[0];
  const int64_t dilation = dilation_spec.get_int_list()[0];

  const auto& in_data = input_spec.get_float_data();
  const auto& w_data = weight_spec.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C * L_out, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t l = 0; l < L_out; ++l) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          const int64_t l_in = l * stride - padding + k * dilation;
          if (l_in >= 0 && l_in < L_in) {
            sum += in_data[n * C * L_in + c * L_in + l_in] * w_data[c * K + k];
          }
        }
        ref_data[n * C * L_out + c * L_out + l] = sum;
      }
    }
  }

  if (!bias_spec.is_none()) {
    const auto& bias_data = bias_spec.get_float_data();
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t l = 0; l < L_out; ++l) {
          ref_data[n * C * L_out + c * L_out + l] += bias_data[c];
        }
      }
    }
  }
}

void conv1d_pw_reference_impl(TestCase& test_case) {
  const auto& input_spec = test_case.inputs()[0];
  const auto& weight_spec = test_case.inputs()[1];
  const auto& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto in_sizes = input_spec.get_tensor_sizes();
  auto w_sizes = weight_spec.get_tensor_sizes();

  int64_t N = in_sizes[0];
  int64_t C_in = in_sizes[1];
  int64_t L = in_sizes[2];
  int64_t C_out = w_sizes[0];

  const auto& in_data = input_spec.get_float_data();
  const auto& w_data = weight_spec.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C_out * L, 0.0f);

  // input is NCHW-contiguous: [N, C_in, L]
  // weight is [C_out, C_in, 1]
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t oc = 0; oc < C_out; ++oc) {
      for (int64_t l = 0; l < L; ++l) {
        float sum = 0.0f;
        for (int64_t ic = 0; ic < C_in; ++ic) {
          sum += in_data[n * C_in * L + ic * L + l] * w_data[oc * C_in + ic];
        }
        ref_data[n * C_out * L + oc * L + l] = sum;
      }
    }
  }

  if (!bias_spec.is_none()) {
    const auto& bias_data = bias_spec.get_float_data();
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t l = 0; l < L; ++l) {
          ref_data[n * C_out * L + oc * L + l] += bias_data[oc];
        }
      }
    }
  }
}

} // namespace conv1d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
