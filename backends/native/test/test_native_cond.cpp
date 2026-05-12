/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Control-flow smoke test for backends/native.
 *
 * Loads CondModel: forward(pred, x) = if pred then x+x else x*x.
 * Runs the same loaded module twice, with pred=True and pred=False,
 * and checks both branches produce the right result.
 *
 * The HOP itself stays at the outer level (executed by ET's standard
 * control-flow); each branch's ops get partitioned into NativeBackend
 * delegate subgraphs via the recursive is_submodule=True path.
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_cond.pte)
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

namespace {

bool run_branch(Module& module, bool pred, float expected) {
  std::vector<float> x_data(2 * 3, 1.0f);
  auto x = from_blob(x_data.data(), {2, 3});

  // pred is a scalar Bool tensor.
  std::vector<uint8_t> pred_data = {static_cast<uint8_t>(pred ? 1 : 0)};
  auto pred_tensor = from_blob(
      pred_data.data(),
      {},
      ::executorch::aten::ScalarType::Bool);

  std::vector<::executorch::runtime::EValue> inputs = {pred_tensor, x};
  auto result = module.forward(inputs);
  if (!result.ok()) {
    fprintf(stderr,
            "ERROR: forward(pred=%d) failed: %d\n",
            pred,
            static_cast<int>(result.error()));
    return false;
  }
  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    fprintf(stderr, "ERROR: pred=%d: expected tensor output\n", pred);
    return false;
  }
  auto out = outputs[0].toTensor();
  const float* p = out.const_data_ptr<float>();
  size_t numel = static_cast<size_t>(out.numel());

  printf("  pred=%s -> output:", pred ? "True " : "False");
  for (size_t i = 0; i < numel; ++i) printf(" %.2f", p[i]);
  printf("  (expected %.2f everywhere)\n", expected);

  for (size_t i = 0; i < numel; ++i) {
    if (std::abs(p[i] - expected) > 1e-5f) {
      fprintf(stderr,
              "ERROR: pred=%d output[%zu]=%.4f, expected %.4f\n",
              pred, i, p[i], expected);
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  const char* env_pte = std::getenv("ET_TESTING_MODEL_PATH");
  std::string pte_path = env_pte
      ? std::string(env_pte)
      : std::string("/tmp/native_cond.pte");

  printf("=== test_native_cond ===\n");
  printf("  Loading: %s\n", pte_path.c_str());

  Module module(pte_path);
  Error load_err = module.load(native_test_util::load_options_for_compute_unit());
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  // pred=True  -> x+x = 2.0
  // pred=False -> x*x = 1.0  (1*1)
  if (!run_branch(module, true, 2.0f)) return 3;
  if (!run_branch(module, false, 1.0f)) return 3;

  printf("=== PASS ===\n");
  return 0;
}
