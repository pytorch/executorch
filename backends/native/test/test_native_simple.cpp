/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Single-shot smoke test: load a .pte and execute one forward pass.
 *
 * Used by backends/native/run_simple_test.sh as a focused smoke test —
 * verifies build + AOT + runtime load + single execute end-to-end. The
 * existing test_model.cpp at the repo root runs a multi-threaded stress
 * test that's better suited to a different kind of validation; this one
 * just answers "does the runtime load and execute one pass?"
 *
 * .pte path: ET_TESTING_MODEL_PATH env var, default /tmp/native_simple.pte
 *
 * Exit status:
 *   0   — load + execute succeeded
 *   1   — load failure
 *   2   — execute failure
 *   3   — output assertion failed
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ::executorch::extension;
using ::executorch::runtime::Error;

int main() {
  const char* env_path = std::getenv("ET_TESTING_MODEL_PATH");
  std::string pte_path =
      env_path ? std::string(env_path) : std::string("/tmp/native_simple.pte");

  printf("=== test_native_simple ===\n");
  printf("  Loading: %s\n", pte_path.c_str());

  Module module(pte_path);

  Error load_err = module.load();
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  // TinyAdd model from export_simple_model.py: forward(a, b) = a + b
  // Both inputs are (1, 4) float32.
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};

  auto a = from_blob(a_data.data(), {1, 4});
  auto b = from_blob(b_data.data(), {1, 4});

  auto result = module.forward({a, b});
  if (!result.ok()) {
    fprintf(stderr,
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
  auto out_tensor = outputs[0].toTensor();
  const float* out_ptr = out_tensor.const_data_ptr<float>();

  printf("  Output: [%.2f, %.2f, %.2f, %.2f]\n",
         out_ptr[0], out_ptr[1], out_ptr[2], out_ptr[3]);
  printf("  Expected: [11.00, 22.00, 33.00, 44.00]\n");

  // Tolerance check on element-wise sum.
  const float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
  for (int i = 0; i < 4; ++i) {
    if (std::abs(out_ptr[i] - expected[i]) > 1e-5f) {
      fprintf(stderr,
              "ERROR: output[%d]=%.6f, expected %.6f (diff %.6f)\n",
              i, out_ptr[i], expected[i],
              std::abs(out_ptr[i] - expected[i]));
      return 3;
    }
  }
  printf("=== PASS ===\n");
  return 0;
}
