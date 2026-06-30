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

// Names mirror test_index.py CONFIGS (self/idx/golden bins written per case).
constexpr const char* kIndexCases[] = {
    "index_n16_m5",
    "index_n8_rev",
    "index_n32_m3",
    "index_n4_rep",
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

std::vector<int32_t> read_i32_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const size_t bytes =
      static_cast<size_t>(f.tellg()) / sizeof(int32_t) * sizeof(int32_t);
  f.seekg(0);
  std::vector<int32_t> data(bytes / sizeof(int32_t));
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(bytes));
  return data;
}

bool run_case(const std::string& dir, const char* name) {
  printf("\n--- Test: %s ---\n", name);
  const std::string base = dir + "/" + name;
  std::vector<float> self_data = read_f32_bin(base + ".self.bin");
  std::vector<int32_t> idx32 = read_i32_bin(base + ".idx.bin");
  std::vector<float> golden = read_f32_bin(base + ".golden.bin");
  if (self_data.empty() || idx32.empty() || golden.empty()) {
    printf("FAIL: could not read self/idx/golden for %s\n", name);
    return false;
  }

  Module module(base + ".pte");
  if (module.load_forward() != Error::Ok) {
    printf("FAIL: could not load %s.pte\n", name);
    return false;
  }

  const int32_t n = static_cast<int32_t>(self_data.size());
  const int32_t m = static_cast<int32_t>(idx32.size());
  auto x = make_tensor_ptr({n}, std::vector<float>(self_data));
  // int64 at the program boundary; copy_inputs narrows to the int32 buffer.
  std::vector<int64_t> idx64(idx32.begin(), idx32.end());
  auto idx = make_tensor_ptr({m}, std::vector<int64_t>(idx64));

  auto result = module.forward({EValue(x), EValue(idx)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  // index.Tensor has exactly one output of shape [num_indices]; fail loud else.
  if (outputs.size() != 1 || !outputs[0].isTensor()) {
    printf("FAIL: expected exactly one tensor output\n");
    return false;
  }
  const auto& out_tensor = outputs[0].toTensor();
  if (out_tensor.dim() != 1 || out_tensor.size(0) != m) {
    printf(
        "FAIL: output shape mismatch (dim %d size0 %d, expected [%d])\n",
        (int)out_tensor.dim(),
        (int)(out_tensor.dim() == 1 ? out_tensor.size(0) : -1),
        m);
    return false;
  }
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
    printf("FAIL: %s exceeds tolerance 1e-3\n", name);
    return false;
  }
  printf("PASS: %s\n", name);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::string dir = "/tmp/index";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_INDEX_DIR")) {
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
  for (const char* name : kIndexCases) {
    ok = run_case(dir, name) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll index tests passed\n");
  return 0;
}
