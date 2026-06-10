// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// fp16 GEMM microbenchmark at Llama 3.1 8B prefill shapes (M=1024):
//   tiled    matmul_vec, Texture3D (production default baseline)
//   coopmat  matmul_coopmat (coopmat_mm.glsl), Buffer — our shader, forced
//            past the desktop-only gate via test_etvk.test_mm "coopmat"
//   coopmat_ref  coopmat_mm_ref (NVIDIA shmem_double_buf4 reference port),
//            Buffer — double-buffered shared memory, subgroup size 32
//
// Apples-to-apples: same shapes, same runtime-mat2 row-major [K,N] fp16
// inputs, same fp32 accumulation + fp16 output, same GPU-timestamp timing.
// Small shapes run a CPU fp32 reference; the M=1024 perf cases skip it.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

struct GemmConfig {
  int64_t M;
  int64_t K;
  int64_t N;
};

// Impl rows benchmarked per shape. The coopmat_ref tile is 128x128 (vs 64x64 for
// coopmat), so correctness shapes must align to 128.
static const std::vector<std::string> kImpls = {"tiled", "coopmat", "coopmat_ref"};

static TestCase make_case(const GemmConfig& cfg, const std::string& impl) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  const utils::StorageType storage =
      (impl == "tiled") ? utils::kTexture3D : utils::kBuffer;
  const std::string storage_str =
      (storage == utils::kTexture3D) ? "Texture3D" : "Buffer";

  TestCase tc;
  tc.set_name(
      "fp16_mm_" + impl + "_M" + std::to_string(cfg.M) + "_K" +
      std::to_string(cfg.K) + "_N" + std::to_string(cfg.N) + "_" + storage_str);

  ValueSpec mat1({cfg.M, cfg.K}, dt, storage, utils::kWidthPacked,
                 DataGenType::RANDOM);
  ValueSpec mat2({cfg.K, cfg.N}, dt, storage, utils::kWidthPacked,
                 DataGenType::RANDOM);
  ValueSpec output({cfg.M, cfg.N}, dt, storage, utils::kWidthPacked,
                   DataGenType::ZEROS);

  if (impl == "coopmat_ref") {
    tc.set_operator_name("etvk.coopmat_mm_ref");
    tc.add_input_spec(mat1);
    tc.add_input_spec(mat2);
  } else {
    // test_etvk.test_mm: mat1, mat2, impl_selector, out. "coopmat" forces
    // add_matmul_coopmat_node (bypasses the is_coopmat_eligible iGPU gate);
    // "tiled" forces the matmul_vec path.
    tc.set_operator_name("test_etvk.test_mm.default");
    tc.add_input_spec(mat1);
    tc.add_input_spec(mat2);
    tc.add_input_spec(ValueSpec::make_string(impl));
  }
  tc.add_output_spec(output);

  // tiled accumulates in fp16 (error grows with K); coopmat/coopmat_ref accumulate
  // in fp32, bounded by fp16 input/output rounding only.
  if (impl == "tiled") {
    tc.set_abs_tolerance(1.0f);
    tc.set_rel_tolerance(1e-1f);
  } else {
    tc.set_abs_tolerance(0.5f);
    tc.set_rel_tolerance(5e-2f);
  }
  tc.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});
  return tc;
}

// CPU fp32 reference from the fp16 inputs; oversized (perf) shapes throw and
// the framework marks them SKIPPED.
static void bench_reference(TestCase& tc) {
  const ValueSpec& a = tc.inputs()[0];
  const ValueSpec& b = tc.inputs()[1];
  ValueSpec& out = tc.outputs()[0];
  const auto as = a.get_tensor_sizes();
  const int64_t M = as[0], K = as[1];
  const int64_t N = out.get_tensor_sizes()[1];
  if (M > 256 || K > 256 || N > 256) {
    throw std::invalid_argument("ref: too big");
  }
  const auto& ah = a.get_half_data();
  const auto& bh = b.get_half_data();
  auto& ref = out.get_ref_float_data();
  ref.resize(M * N);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += half_to_float(ah[m * K + k]) * half_to_float(bh[k * N + n]);
      }
      ref[m * N + n] = acc;
    }
  }
}

// Llama 3.1 8B linear shapes (K,N) at prefill M=1024.
static const std::vector<std::pair<int64_t, int64_t>> kShapes = {
    {4096, 4096}, // q_proj / o_proj
    {4096, 1024}, // k_proj / v_proj (GQA)
    {4096, 14336}, // gate_proj / up_proj
    {14336, 4096}, // down_proj
};
static constexpr int64_t kM = 1024;

std::vector<TestCase> generate_cases() {
  std::vector<TestCase> cases;
  const bool correctness_only =
      std::getenv("COOPMAT_BENCH_CORRECTNESS_ONLY") != nullptr;
  if (!correctness_only) {
    for (const auto& kn : kShapes) {
      for (const auto& impl : kImpls) {
        cases.push_back(make_case({kM, kn.first, kn.second}, impl));
      }
    }
  }
  // Correctness: aligned to the coopmat_ref 128x128 tile (and coopmat's 64/32);
  // the second shape dispatches a 2x2 workgroup grid for coopmat_ref.
  static const std::vector<GemmConfig> kCorrectnessShapes = {
      {128, 64, 128}, {256, 128, 256}};
  for (const auto& cfg : kCorrectnessShapes) {
    for (const auto& impl : kImpls) {
      cases.push_back(make_case(cfg, impl));
    }
  }
  return cases;
}

int64_t flop_calc(const TestCase& tc) {
  const auto& in = tc.inputs()[0].get_tensor_sizes();
  const auto& out = tc.outputs()[0].get_tensor_sizes();
  return 2 * in[0] * in[1] * out[1];
}

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "fp16 GEMM: tiled vs coopmat_mm vs double-buffered reference "
               "(Llama 3.1 8B shapes, M=" << kM << ")" << std::endl;
  print_separator();

  auto results = execute_test_cases(
      generate_cases, flop_calc, "Fp16GemmBench",
      /*warmup=*/3, /*runs=*/5, /*reference=*/bench_reference);

  if (results.size() < kShapes.size() * kImpls.size()) {
    return 0; // correctness-only run
  }
  auto gflops = [](float time_us, int64_t M, int64_t K, int64_t N) -> float {
    return time_us > 0 ? (2.0f * M * N * K) / (time_us * 1e3f) : 0.0f;
  };
  auto gemm_kernel = [](const BenchmarkResult& r) -> std::string {
    std::string name = r.get_kernel_name();
    for (const auto& st : r.get_shader_timings()) {
      if (st.shader_name.find("matmul") != std::string::npos ||
          st.shader_name.find("coopmat") != std::string::npos ||
          st.shader_name.find("coopmat_mm_ref") != std::string::npos) {
        name = st.shader_name;
      }
    }
    return name;
  };

  std::cout << "\n========== SUMMARY: fp16 GEMM GFLOP/s (M=" << kM
            << ") ==========\n";
  std::cout << std::left << std::setw(15) << "shape(K,N)" << std::right
            << std::setw(10) << "tiled" << std::setw(10) << "coopmat"
            << std::setw(12) << "coopmat_ref" << std::setw(10) << "ref/coop"
            << "  kernels\n";
  size_t idx = 0;
  for (const auto& kn : kShapes) {
    const float t_us = results[idx].get_avg_time_us();
    const float c_us = results[idx + 1].get_avg_time_us();
    const float d_us = results[idx + 2].get_avg_time_us();
    const std::string c_kernel = gemm_kernel(results[idx + 1]);
    const std::string d_kernel = gemm_kernel(results[idx + 2]);
    idx += 3;
    const float tiled = gflops(t_us, kM, kn.first, kn.second);
    const float coop = gflops(c_us, kM, kn.first, kn.second);
    const float dbuf = gflops(d_us, kM, kn.first, kn.second);
    std::cout << std::left << std::setw(15)
              << ("(" + std::to_string(kn.first) + "," +
                  std::to_string(kn.second) + ")")
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(10) << tiled << std::setw(10) << coop
              << std::setw(12) << dbuf << std::setw(9)
              << std::setprecision(2) << (coop > 0 ? dbuf / coop : 0.0f)
              << "x  " << c_kernel << " | " << d_kernel << "\n";
  }
  return 0;
}
