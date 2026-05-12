/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Dynamic-shape smoke test for backends/native.
 *
 * Loads DynModel: forward(x) = (x + 1.0) * 2.0, traced with dynamic
 * batch dim B in [1, 8]. Runs forward against the SAME loaded module
 * with batch sizes {1, 3, 5, 8}, all-ones inputs. Each output should
 * be a (B, 4) tensor of all 4.0.
 *
 * Verifies:
 *   - Buffer max-shape allocation at init (sized for batch=8 upper bound).
 *   - bind_inputs accepting smaller-than-max input shapes.
 *   - Per-execute reset of transient bindings.
 *   - Engine reading actual shape via TensorImpl.sizes() and producing
 *     correctly-shaped output.
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_dyn.pte)
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
      : std::string("/tmp/native_dyn.pte");

  printf("=== test_native_dyn_shapes ===\n");
  printf("  Loading: %s\n", pte_path.c_str());

  Module module(pte_path);
  Error load_err = module.load(native_test_util::load_options_for_compute_unit());
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  const std::vector<int> batches = {1, 3, 5, 8};
  for (int B : batches) {
    std::vector<float> input(static_cast<size_t>(B) * 4, 1.0f);
    auto x = from_blob(input.data(), {B, 4});
    std::vector<::executorch::runtime::EValue> inputs = {x};
    auto result = module.forward(inputs);
    if (!result.ok()) {
      fprintf(stderr, "ERROR: forward(B=%d) failed: %d\n",
              B, static_cast<int>(result.error()));
      return 2;
    }
    const auto& outputs = result.get();
    if (outputs.empty() || !outputs[0].isTensor()) {
      fprintf(stderr, "ERROR: B=%d: expected at least one tensor output\n", B);
      return 3;
    }
    auto out = outputs[0].toTensor();
    const float* p = out.const_data_ptr<float>();
    size_t numel = static_cast<size_t>(out.numel());
    auto sizes = out.sizes();

    printf("  B=%d -> sizes=[", B);
    for (size_t i = 0; i < sizes.size(); ++i) {
      printf("%s%d", i == 0 ? "" : ",", static_cast<int>(sizes[i]));
    }
    printf("] (numel=%zu)\n", numel);

    // Expected: every element = (1.0 + 1.0) * 2.0 = 4.0.
    if (sizes.size() != 2 || sizes[0] != B || sizes[1] != 4) {
      fprintf(stderr, "ERROR: B=%d: output shape mismatch\n", B);
      return 3;
    }
    if (numel != static_cast<size_t>(B) * 4) {
      fprintf(stderr, "ERROR: B=%d: numel=%zu, expected %d\n",
              B, numel, B * 4);
      return 3;
    }
    for (size_t i = 0; i < numel; ++i) {
      if (std::abs(p[i] - 4.0f) > 1e-5f) {
        fprintf(stderr, "ERROR: B=%d: output[%zu]=%.4f, expected 4.0\n",
                B, i, p[i]);
        return 3;
      }
    }
  }
  printf("=== PASS ===\n");
  return 0;
}
