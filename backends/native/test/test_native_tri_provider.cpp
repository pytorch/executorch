/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Tri-provider routing test: exercises fake_accel + metal + cpu in a
 * SINGLE plan.
 *
 * Model (from export_tri_model.py): cos((x + 1.0) @ w)
 *   - add → fake_accel (first provider that claims aten::add)
 *   - mm  → metal      (fake_accel doesn't claim mm; metal does)
 *   - cos → cpu        (only cpu has aten::cos)
 *
 * This test ALWAYS forces compute_unit="fake_accel|metal|cpu". The
 * default ("auto") would omit fake_accel and route add to metal,
 * defeating the point.
 *
 * .pte path: ET_TESTING_MODEL_PATH (default /tmp/native_tri.pte)
 * .ref path: NATIVE_TRI_REF_PATH   (default /tmp/native_tri.ref)
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ::executorch::extension;
using ::executorch::runtime::BackendOptions;
using ::executorch::runtime::Error;
using ::executorch::runtime::LoadBackendOptionsMap;

int main() {
  const char* env_pte = std::getenv("ET_TESTING_MODEL_PATH");
  std::string pte_path =
      env_pte ? std::string(env_pte) : std::string("/tmp/native_tri.pte");
  const char* env_ref = std::getenv("NATIVE_TRI_REF_PATH");
  std::string ref_path =
      env_ref ? std::string(env_ref) : std::string("/tmp/native_tri.ref");

  printf("=== test_native_tri_provider ===\n");
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

  // Always-on: opt into all three compute units. fake_accel is opt-in
  // only (auto won't enable it), so this test pins it explicitly.
  LoadBackendOptionsMap backend_opts;
  BackendOptions<2> opts;
  opts.set_option("compute_unit", "fake_accel|metal|cpu");
  if (backend_opts.set_options("NativeBackend", opts.view()) != Error::Ok) {
    fprintf(stderr, "ERROR: set_options failed\n");
    return 1;
  }
  printf("  compute_unit (forced): fake_accel|metal|cpu\n");

  Error load_err = module.load(backend_opts);
  if (load_err != Error::Ok) {
    fprintf(stderr, "ERROR: load() failed: %d\n", static_cast<int>(load_err));
    return 1;
  }
  printf("  Loaded OK\n");

  // Match deterministic inputs from export_tri_model.py.
  std::vector<float> x_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> w_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f,
      0.7f, 0.8f, 0.9f,
      1.0f, 1.1f, 1.2f,
  };
  auto x = from_blob(x_data.data(), {1, 4});
  auto w = from_blob(w_data.data(), {4, 3});

  auto result = module.forward({x, w});
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
  size_t numel = static_cast<size_t>(out_tensor.numel());
  if (numel != ref_count) {
    fprintf(stderr,
            "ERROR: output numel=%zu does not match reference numel=%zu\n",
            numel, ref_count);
    return 3;
  }

  printf("  Output (%zu elems):", numel);
  for (size_t i = 0; i < numel; ++i) printf(" %.4f", out_ptr[i]);
  printf("\n  Reference:        ");
  for (size_t i = 0; i < numel; ++i) printf(" %.4f", ref_data[i]);
  printf("\n");

  const float kTol = 1e-4f;
  for (size_t i = 0; i < numel; ++i) {
    float diff = std::abs(out_ptr[i] - ref_data[i]);
    if (diff > kTol) {
      fprintf(stderr,
              "ERROR: output[%zu]=%.6f, reference %.6f (diff %.6f)\n",
              i, out_ptr[i], ref_data[i], diff);
      return 3;
    }
  }
  printf("=== PASS ===\n");
  return 0;
}
