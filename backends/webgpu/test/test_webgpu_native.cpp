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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

static bool test_single_add(const std::string& model_path) {
  printf("\n--- Test: single add (1024x1024) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int dim = 1024;
  constexpr int size = dim * dim;

  std::vector<float> a_data(size);
  std::vector<float> b_data(size);
  for (int i = 0; i < size; i++) {
    a_data[i] = static_cast<float>(i) * 1.0f;
    b_data[i] = static_cast<float>(i) * 2.0f;
  }

  auto a = make_tensor_ptr({dim, dim}, std::vector<float>(a_data));
  auto b = make_tensor_ptr({dim, dim}, std::vector<float>(b_data));

  auto result = module.forward({EValue(a), EValue(b)});
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
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_error = 0.0f;
  int check_count = std::min(size, 1024);
  for (int i = 0; i < check_count; i++) {
    float expected = a_data[i] + b_data[i];
    float error = std::abs(out_data[i] - expected);
    max_error = std::max(max_error, error);
  }

  printf("Max error: %e (checked %d elements)\n", max_error, check_count);
  if (max_error > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: single add test\n");
  return true;
}

int main(int argc, char** argv) {
  std::string model_path = "webgpu_add_test.pte";
  if (argc > 1) {
    model_path = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_TEST_MODEL")) {
    model_path = env;
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    printf("SKIP: %s\n", e.what());
    return 0;
  }

  set_default_webgpu_context(&ctx);
  printf("WebGPU device acquired (native)\n");

  bool ok = test_single_add(model_path);

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll tests passed\n");
  return 0;
}
