/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * In-place op smoke test for backends/native.
 *
 * Loads InplaceModel (from export_inplace_model.py) which does
 *   y = x.clone(); y.add_(1.0); y.mul_(2.0); y.relu_(); return y
 *
 * Verifies the in-place ops dispatch correctly through CpuOps.cpp's
 * aten::X_ → aten::X.out remap registrations.
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_inplace.pte)
 */

#include <executorch/backends/native/test/test_options_util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ::executorch::extension;
using ::executorch::runtime::Error;

int main() {
  const char* env_pte = std::getenv("ET_TESTING_MODEL_PATH");
  std::string pte_path =
      env_pte ? std::string(env_pte) : std::string("/tmp/native_inplace.pte");

  printf("=== test_native_inplace ===\n");
  printf("  Loading: %s\n", pte_path.c_str());

  Module module(pte_path);
  Error load_err =
      module.load(native_test_util::load_options_for_compute_unit());
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  auto x = from_blob(input.data(), {5});

  std::vector<::executorch::runtime::EValue> inputs = {x};
  auto result = module.forward(inputs);
  if (!result.ok()) {
    fprintf(
        stderr,
        "ERROR: forward() failed: %d\n",
        static_cast<int>(result.error()));
    return 2;
  }
  printf("  forward() OK\n");

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    fprintf(stderr, "ERROR: expected at least one tensor output\n");
    return 3;
  }
  auto out = outputs[0].toTensor();
  const float* p = out.const_data_ptr<float>();
  size_t numel = static_cast<size_t>(out.numel());

  // Trace: x = [-2,-1,0,1,2]; +1 => [-1,0,1,2,3]; *2 => [-2,0,2,4,6];
  // relu => [0,0,2,4,6].
  const float expected[] = {0.0f, 0.0f, 2.0f, 4.0f, 6.0f};

  printf("  Output:  ");
  for (size_t i = 0; i < numel; ++i)
    printf(" %.2f", p[i]);
  printf("\n  Expect:  ");
  for (size_t i = 0; i < numel; ++i)
    printf(" %.2f", expected[i]);
  printf("\n");

  if (numel != 5) {
    fprintf(stderr, "ERROR: numel=%zu, expected 5\n", numel);
    return 3;
  }
  for (size_t i = 0; i < numel; ++i) {
    if (std::abs(p[i] - expected[i]) > 1e-5f) {
      fprintf(
          stderr,
          "ERROR: output[%zu]=%.4f, expected %.4f\n",
          i,
          p[i],
          expected[i]);
      return 3;
    }
  }
  printf("=== PASS ===\n");
  return 0;
}
