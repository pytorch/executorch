/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

struct RmsNormCase {
  const char* name;
  std::array<int32_t, 4> sizes;
};

// Mirrors test_rms_norm.py _CASES; the .py writes per-case .pte/input/golden.
constexpr RmsNormCase kRmsNormCases[] = {
    {"baseline", {1, 1, 7, 896}},
    {"width_eq_wg", {1, 1, 1, 64}},
    {"width_lt_wg", {1, 1, 1, 32}},
    {"width_1", {1, 1, 1, 1}},
    {"width_100", {1, 1, 1, 100}},
    {"width_130", {1, 1, 1, 130}},
    {"rank4_guard", {1, 5, 4, 128}},
    {"many_rows", {1, 1, 1024, 64}},
    {"distinct_rows", {1, 1, 5, 256}},
    {"single_row", {1, 1, 1, 896}},
    {"mixed_sign", {1, 1, 4, 128}},
    {"large_4096", {1, 1, 1, 4096}},
    {"large_8192", {1, 1, 1, 8192}},
    {"weight_zeros_neg", {1, 1, 1, 128}},
};

std::vector<float> read_f32_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

bool run_case(const std::string& dir, const RmsNormCase& tc) {
  printf("\n--- Test: rms_norm[%s] ---\n", tc.name);
  const std::string base = dir + "/" + tc.name;
  std::vector<float> input = read_f32_bin(base + ".input.bin");
  std::vector<float> golden = read_f32_bin(base + ".golden.bin");
  if (input.empty() || golden.empty()) {
    printf("FAIL: could not read input/golden for %s\n", tc.name);
    return false;
  }

  Module module(base + ".pte");
  if (module.load_forward() != Error::Ok) {
    printf("FAIL: could not load %s.pte\n", tc.name);
    return false;
  }

  std::vector<int32_t> sizes(tc.sizes.begin(), tc.sizes.end());
  size_t expected = 1;
  for (int32_t d : tc.sizes) {
    expected *= static_cast<size_t>(d);
  }
  if (input.size() != expected) {
    printf(
        "FAIL: input numel %zu != expected %zu for %s\n",
        input.size(),
        expected,
        tc.name);
    return false;
  }
  auto x = make_tensor_ptr(sizes, std::vector<float>(input));
  auto result = module.forward({EValue(x)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }
  const auto& out_tensor = outputs[0].toTensor();
  if (static_cast<size_t>(out_tensor.numel()) != golden.size()) {
    printf(
        "FAIL: output numel %zu != golden %zu\n",
        (size_t)out_tensor.numel(),
        golden.size());
    return false;
  }
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  for (size_t i = 0; i < golden.size(); i++) {
    const float abs_err = std::abs(out_data[i] - golden[i]);
    max_abs_err = std::max(max_abs_err, abs_err);
    const float denom = std::max(std::abs(golden[i]), 1e-6f);
    max_rel_err = std::max(max_rel_err, abs_err / denom);
  }
  printf(
      "Max abs error: %e   Max rel error: %e (%zu elements)\n",
      max_abs_err,
      max_rel_err,
      golden.size());
  if (max_abs_err > 1e-3f || max_rel_err > 1e-3f) {
    printf("FAIL: rms_norm[%s] exceeds tolerance 1e-3\n", tc.name);
    return false;
  }
  printf("PASS: rms_norm[%s]\n", tc.name);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::string dir = "/tmp/rmsn";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_RMS_NORM_DIR")) {
    dir = env;
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    printf("SKIP: %s\n", e.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);
  printf("WebGPU device acquired (native); case dir: %s\n", dir.c_str());

  bool ok = true;
  for (const auto& tc : kRmsNormCases) {
    ok = run_case(dir, tc) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll rms_norm tests passed\n");
  return 0;
}
