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

struct Case {
  const char* name;
  std::vector<int32_t> sizes;
};

// Mirrors _CASES in test_dispatch_order.py (add-chain or rms_norm+add chain).
const std::vector<Case> kCases = {
    {"single", {16, 16}},
    {"chain3", {64, 64}},
    {"chain5_tiny", {1, 1}},
    {"chain5_wide", {7, 896}},
    {"chain8", {256, 256}},
    {"deep32", {128, 128}},
    {"large_chain", {1024, 1024}},
    {"het_small", {1, 1, 7, 896}},
    {"het_deep", {1, 1, 5, 256}},
};

std::vector<float> read_f32_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const size_t bytes =
      static_cast<size_t>(f.tellg()) / sizeof(float) * sizeof(float);
  f.seekg(0);
  std::vector<float> data(bytes / sizeof(float));
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(bytes));
  return data;
}

bool run_case(const std::string& dir, const Case& tc) {
  printf("\n--- dispatch_order[%s] ---\n", tc.name);
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
  auto x = make_tensor_ptr(tc.sizes, std::vector<float>(input));
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
  // Lenient gate: pass iff abs<=tol OR rel<=tol (near-zero goldens).
  if (max_abs_err > 1e-3f && max_rel_err > 1e-3f) {
    printf("FAIL: dispatch_order[%s] exceeds tolerance 1e-3\n", tc.name);
    return false;
  }
  printf("PASS: dispatch_order[%s]\n", tc.name);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::string dir = "/tmp/dispatch_order";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_DISPATCH_ORDER_DIR")) {
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
  for (const auto& tc : kCases) {
    ok = run_case(dir, tc) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll dispatch_order tests passed\n");
  return 0;
}
