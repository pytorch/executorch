/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Stateful-buffer smoke test for backends/native.
 *
 * Loads a model with a mutable buffer accumulator (from
 * export_stateful_model.py) and runs forward() three times against the
 * SAME loaded module, verifying that the accumulator persists across
 * calls:
 *
 *   call 1 → ones(2,3)
 *   call 2 → twos(2,3)
 *   call 3 → threes(2,3)
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_stateful.pte)
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
  std::string pte_path = env_pte
      ? std::string(env_pte)
      : std::string("/tmp/native_stateful.pte");

  printf("=== test_native_stateful ===\n");
  printf("  Loading: %s\n", pte_path.c_str());

  Module module(pte_path);
  Error load_err = module.load(native_test_util::load_options_for_compute_unit());
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  // Per-call input is ones(2, 3). Same buffer across all calls.
  std::vector<float> ones(2 * 3, 1.0f);
  auto x = from_blob(ones.data(), {2, 3});

  for (int call = 1; call <= 3; ++call) {
    std::vector<::executorch::runtime::EValue> inputs = {x};
    auto result = module.forward(inputs);
    if (!result.ok()) {
      fprintf(stderr,
              "ERROR: forward() call %d failed: %d\n",
              call, static_cast<int>(result.error()));
      return 2;
    }
    const auto& outputs = result.get();
    if (outputs.empty() || !outputs[0].isTensor()) {
      fprintf(stderr, "ERROR: call %d: expected at least one tensor output\n", call);
      return 3;
    }
    auto out = outputs[0].toTensor();
    const float* p = out.const_data_ptr<float>();
    size_t numel = static_cast<size_t>(out.numel());

    printf("  call %d output:", call);
    for (size_t i = 0; i < numel; ++i) printf(" %.2f", p[i]);
    printf("\n");

    // Expect every element == call (1, 2, 3, ...).
    float expected = static_cast<float>(call);
    for (size_t i = 0; i < numel; ++i) {
      if (std::abs(p[i] - expected) > 1e-5f) {
        fprintf(stderr,
                "ERROR: call %d output[%zu]=%.4f, expected %.4f "
                "(mutable buffer not preserved across calls?)\n",
                call, i, p[i], expected);
        return 3;
      }
    }
  }

  printf("=== PASS ===\n");
  return 0;
}
