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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

// Artifacts directory; set from env/argv in main() before RUN_ALL_TESTS().
std::string g_dir; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

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

void run_case(const UpdateCacheCase& tc) {
  Module module(g_dir + "/" + tc.name + ".pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load " << tc.name << ".pte";

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
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ(static_cast<int>(out_tensor.numel()), cnumel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << cnumel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  for (int i = 0; i < cnumel; i++) {
    max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
  }
  // update_cache is a pure scatter copy: the output must be bit-exact.
  EXPECT_EQ(max_abs_err, 0.0f)
      << "update_cache[" << tc.name << "] not bit-exact (max abs error "
      << max_abs_err << ", checked " << cnumel << " elements)";
}

struct ReplayCase {
  const char* name;
  int h;
  int d;
  std::vector<int> seq_lens;
};

// Multi-step advancing-input_pos cache accumulation, mirroring VulkanSDPATest.
void run_replay(const ReplayCase& rc) {
  int cmax = 0;
  for (int s : rc.seq_lens) {
    cmax += s;
  }

  const int cnumel = cmax * rc.h * rc.d;
  std::vector<float> cache(cnumel);
  for (int i = 0; i < cnumel; i++) {
    cache[i] = static_cast<float>(i) + 100.0f;
  }
  std::vector<float> ref(cache);

  int input_pos = 0;
  for (size_t step = 0; step < rc.seq_lens.size(); step++) {
    const int s = rc.seq_lens[step];
    const int vnumel = s * rc.h * rc.d;
    std::vector<float> value(vnumel);
    const float base = static_cast<float>((input_pos + 1) * 1000);
    for (int i = 0; i < vnumel; i++) {
      value[i] = (base + static_cast<float>(i)) * 0.25f;
    }

    const std::string fname = g_dir + "/" + rc.name + "_step" +
        std::to_string(step) + "_S" + std::to_string(s) + "_pos" +
        std::to_string(input_pos) + ".pte";
    Module module(fname);
    ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load " << fname;

    auto v = make_tensor_ptr({1, s, rc.h, rc.d}, std::vector<float>(value));
    auto c = make_tensor_ptr({1, cmax, rc.h, rc.d}, std::vector<float>(cache));
    auto result = module.forward({EValue(v), EValue(c)});
    ASSERT_TRUE(result.ok()) << "forward failed step " << step << " (error "
                             << (int)result.error() << ")";
    const auto& outputs = result.get();
    ASSERT_TRUE(
        !outputs.empty() && outputs[0].isTensor() &&
        static_cast<int>(outputs[0].toTensor().numel()) == cnumel)
        << "bad cache output at step " << step;
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
    // pure scatter copy: must be bit-exact
    EXPECT_EQ(max_abs_err, 0.0f)
        << "step " << step << " (S=" << s << ",pos=" << input_pos
        << "): max abs error " << max_abs_err;
    input_pos += s;
  }
}

struct NegativeCase {
  const char* name;
  const char* guard;
};

// Single-op, single-guard-violation cases: rejection maps to the named guard.
void run_negative_case(const NegativeCase& nc) {
  Module module(g_dir + "/" + nc.name + ".pte");
  const Error err = module.load_forward();
  // init catches the guard throw -> this code; other errors = setup failure.
  EXPECT_EQ(err, Error::DelegateInvalidCompatibility)
      << nc.name << ".pte -> error " << (int)err
      << "; expected DelegateInvalidCompatibility from the '" << nc.guard
      << "' guard";
}

} // namespace

// Single-step scatter cases (prefill / offset / shape variants): the op output
// must equal the inline integer-exact scatter reference.
TEST(UpdateCache, ScatterCases) {
  for (const auto& tc : kCases) {
    run_case(tc);
  }
}

// Multi-step advancing-input_pos cache accumulation, mirroring VulkanSDPATest.
TEST(UpdateCache, Replay) {
  const std::vector<ReplayCase> kReplays = {
      {"seqA", 4, 4, {3, 1, 1, 5, 1, 1, 2}},
      {"seqB", 2, 8, {3, 1, 1, 5, 1, 1}},
      {"llama3", 8, 128, {111, 1, 1, 1, 57, 1, 1}},
  };
  for (const auto& rc : kReplays) {
    run_replay(rc);
  }
}

// Guard-violation cases: each must be rejected with
// DelegateInvalidCompatibility.
TEST(UpdateCache, Negative) {
  const NegativeCase kNegatives[] = {
      {"neg_batch", "batch must be 1"},
      {"neg_fp16", "fp32-only"},
  };
  for (const auto& nc : kNegatives) {
    run_negative_case(nc);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Artifacts dir: env wins, else first positional arg, else default (gtest
  // flags were already stripped by InitGoogleTest above).
  std::string dir = "/tmp/update_cache";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_UPDATE_CACHE_DIR")) {
    dir = env;
  }
  g_dir = dir;

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    std::printf("SKIP: %s\n", e.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
