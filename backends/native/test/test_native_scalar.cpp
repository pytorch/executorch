/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Scalar-add smoke test for backends/native: forward(x) = x + 1.0.
 *
 * Exercises aten::add.Scalar (scalar EValue operand), distinct from
 * aten::add.Tensor used by the simple test. Compares against an
 * exported reference output.
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_scalar.pte)
 * .ref path: NATIVE_SCALAR_REF_PATH (default /tmp/native_scalar.ref)
 */

#include <executorch/backends/native/test/test_options_util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ::executorch::extension;
using ::executorch::runtime::Error;

int main() {
  const char* env_pte = std::getenv("ET_TESTING_MODEL_PATH");
  std::string pte_path =
      env_pte ? std::string(env_pte) : std::string("/tmp/native_scalar.pte");
  const char* env_ref = std::getenv("NATIVE_SCALAR_REF_PATH");
  std::string ref_path =
      env_ref ? std::string(env_ref) : std::string("/tmp/native_scalar.ref");

  printf("=== test_native_scalar ===\n");
  printf("  Loading: %s\n", pte_path.c_str());
  printf("  Reference: %s\n", ref_path.c_str());

  std::ifstream ref_file(ref_path, std::ios::binary | std::ios::ate);
  if (!ref_file) {
    fprintf(stderr, "ERROR: cannot open reference file %s\n", ref_path.c_str());
    return 4;
  }
  std::streamsize ref_bytes = ref_file.tellg();
  if (ref_bytes <= 0 || (ref_bytes % sizeof(float)) != 0) {
    fprintf(stderr, "ERROR: reference file has invalid size\n");
    return 4;
  }
  ref_file.seekg(0, std::ios::beg);
  size_t ref_count = static_cast<size_t>(ref_bytes) / sizeof(float);
  std::vector<float> ref_data(ref_count);
  ref_file.read(reinterpret_cast<char*>(ref_data.data()), ref_bytes);

  Module module(pte_path);
  Error load_err =
      module.load(native_test_util::load_options_for_compute_unit());
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  // Match deterministic input from export_scalar_model.py:
  // torch.tensor([[1.0, 2.0, 3.0, 4.0]]).
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  auto x = from_blob(input.data(), {1, 4});

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
  auto out_tensor = outputs[0].toTensor();
  const float* out_ptr = out_tensor.const_data_ptr<float>();
  size_t numel = static_cast<size_t>(out_tensor.numel());
  if (numel != ref_count) {
    fprintf(
        stderr,
        "ERROR: output numel=%zu does not match reference numel=%zu\n",
        numel,
        ref_count);
    return 3;
  }

  printf("  Output (%zu elems):", numel);
  for (size_t i = 0; i < numel; ++i)
    printf(" %.4f", out_ptr[i]);
  printf("\n  Reference:        ");
  for (size_t i = 0; i < numel; ++i)
    printf(" %.4f", ref_data[i]);
  printf("\n");

  const float kTol = 1e-5f;
  for (size_t i = 0; i < numel; ++i) {
    float diff = std::abs(out_ptr[i] - ref_data[i]);
    if (diff > kTol) {
      fprintf(
          stderr,
          "ERROR: output[%zu]=%.6f, reference %.6f (diff %.6f)\n",
          i,
          out_ptr[i],
          ref_data[i],
          diff);
      return 3;
    }
  }
  printf("=== PASS ===\n");
  return 0;
}
