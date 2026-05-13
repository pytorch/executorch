// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 256;

// Return the input data as float regardless of dtype. For kHalf, the
// underlying half_data is converted via half_to_float; for kFloat the
// existing float_data is copied. The CPU reference computes in float and
// the framework compares against (and converts back to) the GPU output.
static std::vector<float> as_float_data(const ValueSpec& spec) {
  if (spec.dtype == vkapi::kFloat) {
    return spec.get_float_data();
  }
  if (spec.dtype == vkapi::kHalf) {
    const auto& half_bits = spec.get_half_data();
    std::vector<float> out(half_bits.size());
    for (size_t i = 0; i < half_bits.size(); ++i) {
      out[i] = half_to_float(half_bits[i]);
    }
    return out;
  }
  throw std::invalid_argument("as_float_data: unsupported dtype");
}

struct MmConfig {
  // mat1: [M, K] or [B, M, K]
  // mat2: [K, N] or [B, K, N]
  int64_t B; // batch size, 0 for non-batched
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias; // true for addmm/linear
  bool mat2_is_transposed; // true for linear (weight is [N, K])
  bool mat2_is_constant; // true to test prepacked linear path
  // "default" routes through aten.{mm,linear,...}.default (production path).
  // "coopmat" / "tiled" force-dispatch a specific shader implementation.
  std::string impl_selector = "default";
};

struct MmShape {
  int64_t B, M, K, N;
};

static TestCase create_mm_test_case(
    const MmConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout) {
  TestCase test_case;

  bool is_batched = config.B > 0;
  bool is_perf = config.M > kRefDimSizeLimit || config.K > kRefDimSizeLimit ||
      config.N > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type);
  std::string layout_str = layout_abbrev(memory_layout);
  std::string dtype_str = (dtype == vkapi::kHalf) ? "f16" : "f32";

  // Determine op type string
  std::string op_type;
  if (config.mat2_is_transposed) {
    op_type = config.has_bias ? "linear-bias" : "linear";
  } else if (config.has_bias) {
    op_type = config.mat2_is_constant ? "addmm-const-mat2" : "addmm";
  } else if (is_batched) {
    op_type = config.mat2_is_constant ? "bmm-const-mat2" : "bmm";
  } else {
    op_type = config.mat2_is_constant ? "mm-const-mat2" : "mm";
  }

  // Build shape string
  std::string shape;
  if (is_batched) {
    shape = "[" + std::to_string(config.B) + "," + std::to_string(config.M) +
        "," + std::to_string(config.K) + "]x[" + std::to_string(config.B) +
        "," + std::to_string(config.K) + "," + std::to_string(config.N) + "]";
  } else if (config.mat2_is_transposed) {
    shape = "[" + std::to_string(config.M) + "," + std::to_string(config.K) +
        "]x[" + std::to_string(config.N) + "," + std::to_string(config.K) + "]";
  } else {
    shape = "[" + std::to_string(config.M) + "," + std::to_string(config.K) +
        "]x[" + std::to_string(config.K) + "," + std::to_string(config.N) + "]";
  }

  std::string name = prefix + "  " + op_type + " " + shape + "  " +
      storage_str + "(" + layout_str + ") " + dtype_str;
  if (config.impl_selector != "default") {
    name += " [" + config.impl_selector + "]";
  }

  test_case.set_name(name);

  // Determine op name - use test wrapper operators
  std::string op_name;
  if (is_batched) {
    op_name = "test_etvk.test_bmm.default";
  } else if (config.mat2_is_transposed) {
    op_name = "test_etvk.test_linear.default";
  } else if (config.has_bias) {
    op_name = "test_etvk.test_addmm.default";
  } else {
    op_name = "test_etvk.test_mm.default";
  }
  test_case.set_operator_name(op_name);

  // mat1
  std::vector<int64_t> mat1_sizes;
  if (is_batched) {
    mat1_sizes = {config.B, config.M, config.K};
  } else {
    mat1_sizes = {config.M, config.K};
  }
  ValueSpec mat1(
      mat1_sizes, dtype, storage_type, memory_layout, DataGenType::RANDOM);

  // mat2 - for linear, weight is [N, K] (transposed)
  std::vector<int64_t> mat2_sizes;
  if (config.mat2_is_transposed) {
    mat2_sizes = {config.N, config.K};
  } else if (is_batched) {
    mat2_sizes = {config.B, config.K, config.N};
  } else {
    mat2_sizes = {config.K, config.N};
  }

  if (config.mat2_is_transposed) {
    // For linear, weight is a constant tensor
    ValueSpec mat2(
        mat2_sizes, dtype, storage_type, memory_layout, DataGenType::RANDOM);
    mat2.set_constant(true);

    // bias (or none)
    if (config.has_bias) {
      ValueSpec bias(
          {config.N}, dtype, storage_type, memory_layout, DataGenType::RANDOM);
      bias.set_constant(true);

      // test_etvk.test_linear.default: input, weight, bias, impl_selector, out
      test_case.add_input_spec(mat1);
      test_case.add_input_spec(mat2);
      test_case.add_input_spec(bias);
    } else {
      test_case.add_input_spec(mat1);
      test_case.add_input_spec(mat2);
      // Use an int spec marked as none to avoid being treated as a tensor
      ValueSpec none_bias(static_cast<int32_t>(0));
      none_bias.set_none(true);
      test_case.add_input_spec(none_bias);
    }
  } else if (config.has_bias) {
    // test_etvk.test_addmm.default: self, mat1, mat2, beta, alpha,
    // impl_selector, out
    ValueSpec bias(
        {config.N}, dtype, storage_type, memory_layout, DataGenType::RANDOM);
    ValueSpec mat2(
        mat2_sizes, dtype, storage_type, memory_layout, DataGenType::RANDOM);
    if (config.mat2_is_constant) {
      mat2.set_constant(true);
    }

    test_case.add_input_spec(bias);
    test_case.add_input_spec(mat1);
    test_case.add_input_spec(mat2);
    // beta
    test_case.add_input_spec(ValueSpec(1.0f));
    // alpha
    test_case.add_input_spec(ValueSpec(1.0f));
  } else {
    // test_etvk.test_mm.default or test_etvk.test_bmm.default:
    // mat1, mat2, impl_selector, out
    ValueSpec mat2(
        mat2_sizes, dtype, storage_type, memory_layout, DataGenType::RANDOM);
    if (config.mat2_is_constant) {
      mat2.set_constant(true);
    }
    test_case.add_input_spec(mat1);
    test_case.add_input_spec(mat2);
  }

  // impl_selector (added before output for all variants)
  ValueSpec impl_selector_spec = ValueSpec::make_string(config.impl_selector);
  test_case.add_input_spec(impl_selector_spec);

  // output
  std::vector<int64_t> out_sizes;
  if (is_batched) {
    out_sizes = {config.B, config.M, config.N};
  } else {
    out_sizes = {config.M, config.N};
  }
  ValueSpec output(
      out_sizes, dtype, storage_type, memory_layout, DataGenType::ZEROS);
  test_case.add_output_spec(output);

  // The coopmat shader uses fp16 intermediates regardless of input dtype
  // (inputs are packHalf2x16-converted before entering the MMA), so the
  // achievable precision is fp16-bounded for any path that dispatches to
  // it. A coopmat dispatch occurs when impl_selector forces it, or when
  // the default routing's gate (buffer + M/N/K alignment) is met.
  bool routes_to_coopmat = false;
  if (config.impl_selector == "coopmat") {
    routes_to_coopmat = true;
  } else if (
      config.impl_selector == "default" && storage_type == utils::kBuffer &&
      !is_batched && config.M % 64 == 0 && config.N % 64 == 0 &&
      config.K % 32 == 0) {
    // Mirror is_coopmat_eligible's full device-capability gate so default
    // routing's tolerance matches its actual dispatch path. On a device
    // that's integrated, subgroup-32, or otherwise ineligible, the default
    // routes to tiled and tolerance must stay tight.
    const auto* adapter = api::context()->adapter_ptr();
    routes_to_coopmat = adapter->supports_cooperative_matrix() &&
        adapter->subgroup_size() == 64 && !adapter->is_integrated_gpu();
  }

  if (dtype == vkapi::kHalf) {
    // Pure-fp16 GEMM accumulates K products in fp16 (no fp32 accumulator on
    // the tiled path), so error scales with ULP * sqrt(K) * output magnitude.
    // For K up to 4096 in these tests with inputs in [-1, 1], that lands
    // around 6% relative error empirically; 10% covers worst-case alignment
    // of rounding signs without masking real shader bugs.
    test_case.set_abs_tolerance(1.0f);
    test_case.set_rel_tolerance(1e-1f);
  } else if (routes_to_coopmat) {
    // Coopmat uses fp16 inputs with an fp32 accumulator, so error is
    // bounded by fp16 input-conversion noise, not accumulation drift.
    test_case.set_abs_tolerance(1e-1f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  // Filter out layout conversion shaders from timing
  test_case.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});

  return test_case;
}

// Reference implementation for mm/bmm
// Input layout per test operator:
//   test_mm/test_bmm: mat1[0], mat2[1], impl_selector[2]
//   test_addmm: self[0], mat1[1], mat2[2], beta[3], alpha[4], impl_selector[5]
//   test_linear: input[0], weight[1], bias[2], impl_selector[3]
static void mm_reference_impl(TestCase& test_case) {
  const std::string& op_name = test_case.operator_name();
  ValueSpec& output = test_case.outputs()[0];
  auto out_sizes = output.get_tensor_sizes();

  const auto in_dtype = test_case.inputs()[0].dtype;
  if (in_dtype != vkapi::kFloat && in_dtype != vkapi::kHalf) {
    throw std::invalid_argument("Reference only supports float / half");
  }

  if (op_name == "test_etvk.test_mm.default") {
    // mat1[0], mat2[1], impl_selector[2]
    const auto& mat1 = test_case.inputs()[0];
    const auto& mat2 = test_case.inputs()[1];
    auto mat1_sizes = mat1.get_tensor_sizes();
    auto mat2_sizes = mat2.get_tensor_sizes();

    int64_t M = mat1_sizes[0];
    int64_t K = mat1_sizes[1];
    int64_t N = mat2_sizes[1];

    auto mat1_data = as_float_data(mat1);
    auto mat2_data = as_float_data(mat2);
    auto& ref_data = output.get_ref_float_data();
    ref_data.resize(M * N, 0.0f);

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += mat1_data[m * K + k] * mat2_data[k * N + n];
        }
        ref_data[m * N + n] = sum;
      }
    }
  } else if (op_name == "test_etvk.test_bmm.default") {
    // mat1[0], mat2[1], impl_selector[2]
    const auto& mat1 = test_case.inputs()[0];
    const auto& mat2 = test_case.inputs()[1];
    auto mat1_sizes = mat1.get_tensor_sizes();
    auto mat2_sizes = mat2.get_tensor_sizes();

    int64_t B = mat1_sizes[0];
    int64_t M = mat1_sizes[1];
    int64_t K = mat1_sizes[2];
    int64_t N = mat2_sizes[2];

    auto mat1_data = as_float_data(mat1);
    auto mat2_data = as_float_data(mat2);
    auto& ref_data = output.get_ref_float_data();
    ref_data.resize(B * M * N, 0.0f);

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            sum += mat1_data[b * M * K + m * K + k] *
                mat2_data[b * K * N + k * N + n];
          }
          ref_data[b * M * N + m * N + n] = sum;
        }
      }
    }
  } else if (op_name == "test_etvk.test_addmm.default") {
    // self[0], mat1[1], mat2[2], beta[3], alpha[4], impl_selector[5]
    const auto& bias = test_case.inputs()[0];
    const auto& mat1 = test_case.inputs()[1];
    const auto& mat2 = test_case.inputs()[2];
    auto mat1_sizes = mat1.get_tensor_sizes();
    auto mat2_sizes = mat2.get_tensor_sizes();

    int64_t M = mat1_sizes[0];
    int64_t K = mat1_sizes[1];
    int64_t N = mat2_sizes[1];

    auto bias_data = as_float_data(bias);
    auto mat1_data = as_float_data(mat1);
    auto mat2_data = as_float_data(mat2);
    auto& ref_data = output.get_ref_float_data();
    ref_data.resize(M * N, 0.0f);

    float alpha = test_case.inputs()[4].get_float_value();
    float beta = test_case.inputs()[3].get_float_value();

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += mat1_data[m * K + k] * mat2_data[k * N + n];
        }
        float bias_val =
            (n < static_cast<int64_t>(bias_data.size())) ? bias_data[n] : 0.0f;
        ref_data[m * N + n] = beta * bias_val + alpha * sum;
      }
    }
  } else if (op_name == "test_etvk.test_linear.default") {
    // input[0], weight[1], bias[2], impl_selector[3]
    const auto& input = test_case.inputs()[0];
    const auto& weight = test_case.inputs()[1];
    const auto& bias_spec = test_case.inputs()[2];
    auto input_sizes = input.get_tensor_sizes();
    auto weight_sizes = weight.get_tensor_sizes();

    int64_t M = input_sizes[0];
    int64_t K = input_sizes[1];
    int64_t N = weight_sizes[0];

    auto input_data = as_float_data(input);
    auto weight_data = as_float_data(weight);
    std::vector<float> bias_data;
    if (!bias_spec.is_none()) {
      bias_data = as_float_data(bias_spec);
    }
    auto& ref_data = output.get_ref_float_data();
    ref_data.resize(M * N, 0.0f);

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += input_data[m * K + k] * weight_data[n * K + k];
        }
        if (!bias_data.empty()) {
          sum += bias_data[n];
        }
        ref_data[m * N + n] = sum;
      }
    }
  }
}

static std::vector<TestCase> generate_mm_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  std::vector<MmShape> shapes = {
      // Accuracy shapes (float)
      {0, 64, 128, 64},
      {0, 128, 256, 128},
      {0, 32, 64, 256},
      {1, 64, 128, 64},
      {1, 4, 32, 16},
      // Non-multiple-of-4 accuracy shapes (exercises scalar shader fallback)
      {0, 57, 131, 43},
      {0, 33, 67, 91},
      {1, 19, 53, 37},
      {0, 64, 128, 47}, // only N unaligned
      {0, 64, 47, 128}, // only K unaligned
      // Performance shapes (half)
      {0, 4096, 1024, 256},
      {0, 4096, 256, 128},
      {0, 4096, 128, 256},
      {1, 4096, 256, 1024},
      {1, 4096, 256, 128},
      {1, 256, 4096, 64},
      {0, 4096, 64, 128},
  };

  // Coopmat shader requires M%64==0, N%64==0, K%32==0 (no partial-tile or
  // K-tail handling). Sweep both "coopmat" and "tiled" force-dispatch variants
  // for shapes that satisfy alignment, on buffer storage only (coopmat
  // requires buffer outputs).
  auto coopmat_shape_eligible = [](const MmShape& s) {
    return s.B == 0 && s.M % 64 == 0 && s.N % 64 == 0 && s.K % 32 == 0;
  };

  // The "coopmat" forced-dispatch sweep skips the runtime eligibility check
  // (it's intentionally bypassing the gate to exercise the shader). Skip
  // generating those cases on adapters that can't actually run the shader,
  // or pipeline creation/dispatch will fail.
  const auto* adapter = api::context()->adapter_ptr();
  const bool coopmat_runnable =
      adapter->supports_cooperative_matrix() && adapter->subgroup_size() == 64;
  std::vector<std::string> coopmat_sweep_selectors = {"tiled"};
  if (coopmat_runnable) {
    coopmat_sweep_selectors.push_back("coopmat");
  }

  for (const auto& s : shapes) {
    bool is_batched = s.B > 0;
    bool is_perf = s.M > kRefDimSizeLimit || s.K > kRefDimSizeLimit ||
        s.N > kRefDimSizeLimit;

    std::vector<vkapi::ScalarType> dtypes = is_perf
        ? std::vector<vkapi::ScalarType>{vkapi::kFloat, vkapi::kHalf}
        : std::vector<vkapi::ScalarType>{vkapi::kFloat};

    MmConfig dynamic_cfg{s.B, s.M, s.K, s.N, false, false, false};
    MmConfig const_cfg{s.B, s.M, s.K, s.N, false, false, true};

    for (auto dtype : dtypes) {
      for (auto st : storage_types) {
        test_cases.push_back(
            create_mm_test_case(dynamic_cfg, dtype, st, utils::kWidthPacked));
        test_cases.push_back(
            create_mm_test_case(const_cfg, dtype, st, utils::kWidthPacked));
      }

      // Coopmat A/B sweep: only on aligned shapes + buffer storage.
      if (coopmat_shape_eligible(s)) {
        for (const auto& sel : coopmat_sweep_selectors) {
          MmConfig dyn = dynamic_cfg;
          dyn.impl_selector = sel;
          test_cases.push_back(create_mm_test_case(
              dyn, dtype, utils::kBuffer, utils::kWidthPacked));
          MmConfig con = const_cfg;
          con.impl_selector = sel;
          test_cases.push_back(create_mm_test_case(
              con, dtype, utils::kBuffer, utils::kWidthPacked));
        }
      }

      if (!is_batched) {
        MmConfig addmm_cfg{s.B, s.M, s.K, s.N, true, false, false};
        MmConfig addmm_const_cfg{s.B, s.M, s.K, s.N, true, false, true};
        MmConfig linear_cfg{s.B, s.M, s.K, s.N, false, true, false};
        MmConfig linear_bias_cfg{s.B, s.M, s.K, s.N, true, true, false};

        for (auto st : storage_types) {
          test_cases.push_back(
              create_mm_test_case(addmm_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(create_mm_test_case(
              addmm_const_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(
              create_mm_test_case(linear_cfg, dtype, st, utils::kWidthPacked));
          test_cases.push_back(create_mm_test_case(
              linear_bias_cfg, dtype, st, utils::kWidthPacked));
        }

        // Coopmat A/B sweep on linear paths too (only aligned shapes).
        if (coopmat_shape_eligible(s)) {
          for (const auto& sel : coopmat_sweep_selectors) {
            MmConfig lin = linear_cfg;
            lin.impl_selector = sel;
            test_cases.push_back(create_mm_test_case(
                lin, dtype, utils::kBuffer, utils::kWidthPacked));
            MmConfig lin_bias = linear_bias_cfg;
            lin_bias.impl_selector = sel;
            test_cases.push_back(create_mm_test_case(
                lin_bias, dtype, utils::kBuffer, utils::kWidthPacked));
          }
        }
      }
    }
  }

  return test_cases;
}

static int64_t mm_flop_calculator(const TestCase& test_case) {
  const auto& out_sizes = test_case.outputs()[0].get_tensor_sizes();
  const std::string& op_name = test_case.operator_name();

  int64_t M, N, K;

  if (op_name == "test_etvk.test_mm.default") {
    // mat1[0], mat2[1], impl_selector[2]
    auto mat1_sizes = test_case.inputs()[0].get_tensor_sizes();
    M = mat1_sizes[0];
    K = mat1_sizes[1];
    N = out_sizes[1];
    return 2 * M * K * N;
  } else if (op_name == "test_etvk.test_bmm.default") {
    // mat1[0], mat2[1], impl_selector[2]
    auto mat1_sizes = test_case.inputs()[0].get_tensor_sizes();
    int64_t B = mat1_sizes[0];
    M = mat1_sizes[1];
    K = mat1_sizes[2];
    N = out_sizes[2];
    return 2 * B * M * K * N;
  } else if (op_name == "test_etvk.test_addmm.default") {
    // self[0], mat1[1], mat2[2], beta[3], alpha[4], impl_selector[5]
    auto mat1_sizes = test_case.inputs()[1].get_tensor_sizes();
    M = mat1_sizes[0];
    K = mat1_sizes[1];
    N = out_sizes[1];
    return 2 * M * K * N;
  } else if (op_name == "test_etvk.test_linear.default") {
    // input[0], weight[1], bias[2], impl_selector[3]
    auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
    auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();
    M = input_sizes[0];
    K = input_sizes[1];
    N = weight_sizes[0];
    return 2 * M * K * N;
  }
  return 1;
}

static void reference_impl(TestCase& test_case) {
  mm_reference_impl(test_case);
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Matrix Multiply (mm/bmm/addmm/linear) Benchmark" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_mm_test_cases,
      mm_flop_calculator,
      "MatMul",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      ref_fn);

  return 0;
}
