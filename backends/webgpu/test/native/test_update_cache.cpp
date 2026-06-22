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
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

struct UpdateCacheCase {
  const char* name;
  int s;
  int h;
  int d;
  int cmax;
  int input_pos;
};

// Mirrors test_update_cache.py CASES; golden scatter is integer-exact (inline).
constexpr UpdateCacheCase kCases[] = {
    {"prefill", 2, 2, 4, 8, 0},
    {"offset", 2, 2, 4, 8, 5},
    {"shape_b", 3, 4, 8, 16, 0},
    {"shape_b_offset", 3, 4, 8, 16, 10},
};

bool run_case(const std::string& dir, const UpdateCacheCase& tc) {
  printf(
      "\n--- Test: update_cache[%s] (S=%d,H=%d,D=%d,Cmax=%d,pos=%d) ---\n",
      tc.name,
      tc.s,
      tc.h,
      tc.d,
      tc.cmax,
      tc.input_pos);
  Module module(dir + "/" + tc.name + ".pte");
  if (module.load_forward() != Error::Ok) {
    printf("FAIL: could not load %s.pte\n", tc.name);
    return false;
  }

  const int vnumel = tc.s * tc.h * tc.d;
  const int cnumel = tc.cmax * tc.h * tc.d;
  std::vector<float> value(vnumel);
  std::vector<float> cache(cnumel);
  for (int i = 0; i < vnumel; i++) {
    value[i] = static_cast<float>(i) * 0.5f;
  }
  for (int i = 0; i < cnumel; i++) {
    cache[i] = static_cast<float>(i) + 100.0f;
  }

  // Inline reference: scatter value into the cache at input_pos, bounds-checked
  // exactly as the op (integer-exact copy, no library needed).
  std::vector<float> ref(cache);
  const int dst_offset = tc.input_pos * tc.h * tc.d;
  for (int i = 0; i < vnumel; i++) {
    if (dst_offset + i < cnumel) {
      ref[dst_offset + i] = value[i];
    }
  }

  auto v = make_tensor_ptr({1, tc.s, tc.h, tc.d}, std::vector<float>(value));
  auto c = make_tensor_ptr({1, tc.cmax, tc.h, tc.d}, std::vector<float>(cache));
  auto result = module.forward({EValue(v), EValue(c)});
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
  if (static_cast<int>(out_tensor.numel()) != cnumel) {
    printf(
        "FAIL: output numel %zu != expected %d\n",
        (size_t)out_tensor.numel(),
        cnumel);
    return false;
  }
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  for (int i = 0; i < cnumel; i++) {
    max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
  }
  printf("Max abs error: %e (checked %d elements)\n", max_abs_err, cnumel);
  // update_cache is a pure scatter copy: the output must be bit-exact.
  if (max_abs_err > 0.0f) {
    printf("FAIL: update_cache[%s] not bit-exact\n", tc.name);
    return false;
  }
  printf("PASS: update_cache[%s]\n", tc.name);
  return true;
}

struct ReplayCase {
  const char* name;
  int h;
  int d;
  std::vector<int> seq_lens;
};

// Multi-step advancing-input_pos cache accumulation, mirroring VulkanSDPATest.
bool run_replay(const std::string& dir, const ReplayCase& rc) {
  int cmax = 0;
  for (int s : rc.seq_lens) {
    cmax += s;
  }
  printf(
      "\n--- Replay: update_cache[%s] (H=%d,D=%d,Cmax=%d,%zu steps) ---\n",
      rc.name,
      rc.h,
      rc.d,
      cmax,
      rc.seq_lens.size());

  const int cnumel = cmax * rc.h * rc.d;
  std::vector<float> cache(cnumel);
  for (int i = 0; i < cnumel; i++) {
    cache[i] = static_cast<float>(i) + 100.0f;
  }
  std::vector<float> ref(cache);

  int input_pos = 0;
  bool ok = true;
  for (size_t step = 0; step < rc.seq_lens.size(); step++) {
    const int s = rc.seq_lens[step];
    const int vnumel = s * rc.h * rc.d;
    std::vector<float> value(vnumel);
    const float base = static_cast<float>((input_pos + 1) * 1000);
    for (int i = 0; i < vnumel; i++) {
      value[i] = (base + static_cast<float>(i)) * 0.25f;
    }

    const std::string fname = dir + "/" + rc.name + "_step" +
        std::to_string(step) + "_S" + std::to_string(s) + "_pos" +
        std::to_string(input_pos) + ".pte";
    Module module(fname);
    if (module.load_forward() != Error::Ok) {
      printf("FAIL: could not load %s\n", fname.c_str());
      return false;
    }

    auto v = make_tensor_ptr({1, s, rc.h, rc.d}, std::vector<float>(value));
    auto c = make_tensor_ptr({1, cmax, rc.h, rc.d}, std::vector<float>(cache));
    auto result = module.forward({EValue(v), EValue(c)});
    if (!result.ok()) {
      printf(
          "FAIL: forward failed step %zu (error %d)\n",
          step,
          (int)result.error());
      return false;
    }
    const auto& outputs = result.get();
    if (outputs.empty() || !outputs[0].isTensor() ||
        static_cast<int>(outputs[0].toTensor().numel()) != cnumel) {
      printf("FAIL: bad cache output at step %zu\n", step);
      return false;
    }
    const float* out_data = outputs[0].toTensor().const_data_ptr<float>();

    const int dst_offset = input_pos * rc.h * rc.d;
    for (int i = 0; i < vnumel; i++) {
      if (dst_offset + i < cnumel) {
        ref[dst_offset + i] = value[i];
      }
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < cnumel; i++) {
      max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
      cache[i] = out_data[i]; // thread the accumulated cache into the next step
    }
    printf(
        "  step %zu (S=%d,pos=%d): max abs error %e\n",
        step,
        s,
        input_pos,
        max_abs_err);
    if (max_abs_err > 0.0f) { // pure scatter copy: must be bit-exact
      ok = false;
    }
    input_pos += s;
  }

  if (ok) {
    printf("PASS: update_cache[%s] replay\n", rc.name);
  } else {
    printf("FAIL: update_cache[%s] replay\n", rc.name);
  }
  return ok;
}

struct NegativeCase {
  const char* name;
  const char* guard;
};

// Single-op, single-guard-violation cases: rejection maps to the named guard.
bool run_negative_case(const std::string& dir, const NegativeCase& nc) {
  printf(
      "\n--- Negative: update_cache[%s] (expect rejection: %s) ---\n",
      nc.name,
      nc.guard);
  Module module(dir + "/" + nc.name + ".pte");
  const Error err = module.load_forward();
  // init catches the guard throw -> this code; other errors = setup failure.
  if (err != Error::DelegateInvalidCompatibility) {
    printf(
        "FAIL: %s.pte -> error %d; expected DelegateInvalidCompatibility "
        "from the '%s' guard\n",
        nc.name,
        (int)err,
        nc.guard);
    return false;
  }
  printf("PASS: rejected with DelegateInvalidCompatibility (%s)\n", nc.guard);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::string dir = "/tmp/update_cache";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_UPDATE_CACHE_DIR")) {
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

  const std::vector<ReplayCase> kReplays = {
      {"seqA", 4, 4, {3, 1, 1, 5, 1, 1, 2}},
      {"seqB", 2, 8, {3, 1, 1, 5, 1, 1}},
      {"llama3", 8, 128, {111, 1, 1, 1, 57, 1, 1}},
  };
  for (const auto& rc : kReplays) {
    ok = run_replay(dir, rc) && ok;
  }

  const NegativeCase kNegatives[] = {
      {"neg_batch", "batch must be 1"},
      {"neg_fp16", "fp32-only"},
  };
  for (const auto& nc : kNegatives) {
    ok = run_negative_case(dir, nc) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll update_cache tests passed\n");
  return 0;
}
