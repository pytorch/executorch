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
#include <fstream>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

// Artifacts directory; set from env/argv in main() before RUN_ALL_TESTS().
std::string g_dir; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

std::vector<float> read_f32_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const auto file_size = static_cast<size_t>(f.tellg());
  if (file_size % sizeof(float) != 0) {
    return {}; // truncated/corrupt golden; caller treats empty as failure
  }
  f.seekg(0);
  std::vector<float> data(file_size / sizeof(float));
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(file_size));
  return data;
}

// Mirrors _CASES in test_dispatch_order.py (add-chain or rms_norm+add chain).
void run_case(const char* name, const std::vector<int32_t>& sizes) {
  const std::string base = g_dir + "/" + name;
  std::vector<float> input = read_f32_bin(base + ".input.bin");
  std::vector<float> golden = read_f32_bin(base + ".golden.bin");
  ASSERT_FALSE(input.empty() || golden.empty())
      << "could not read input/golden for " << name;

  Module module(base + ".pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load " << name << ".pte";

  size_t expected = 1;
  for (int32_t d : sizes) {
    expected *= static_cast<size_t>(d);
  }
  ASSERT_EQ(input.size(), expected)
      << "input numel " << input.size() << " != expected " << expected
      << " for " << name;
  auto x = make_tensor_ptr(sizes, std::vector<float>(input));
  auto result = module.forward({EValue(x)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ(static_cast<size_t>(out_tensor.numel()), golden.size())
      << "output numel " << (size_t)out_tensor.numel() << " != golden "
      << golden.size();
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  for (size_t i = 0; i < golden.size(); i++) {
    const float abs_err = std::abs(out_data[i] - golden[i]);
    max_abs_err = std::max(max_abs_err, abs_err);
    const float denom = std::max(std::abs(golden[i]), 1e-6f);
    max_rel_err = std::max(max_rel_err, abs_err / denom);
  }
  // Lenient gate: pass iff abs<=tol OR rel<=tol (near-zero goldens).
  EXPECT_FALSE(max_abs_err > 1e-3f && max_rel_err > 1e-3f)
      << "dispatch_order[" << name
      << "] exceeds tolerance 1e-3 (max_abs_err=" << max_abs_err
      << " max_rel_err=" << max_rel_err << ", " << golden.size()
      << " elements)";
}

} // namespace

TEST(DispatchOrder, single) {
  run_case("single", {16, 16});
}

TEST(DispatchOrder, chain3) {
  run_case("chain3", {64, 64});
}

TEST(DispatchOrder, chain5_tiny) {
  run_case("chain5_tiny", {1, 1});
}

TEST(DispatchOrder, chain5_wide) {
  run_case("chain5_wide", {7, 896});
}

TEST(DispatchOrder, chain8) {
  run_case("chain8", {256, 256});
}

TEST(DispatchOrder, deep32) {
  run_case("deep32", {128, 128});
}

TEST(DispatchOrder, large_chain) {
  run_case("large_chain", {1024, 1024});
}

TEST(DispatchOrder, het_small) {
  run_case("het_small", {1, 1, 7, 896});
}

TEST(DispatchOrder, het_deep) {
  run_case("het_deep", {1, 1, 5, 256});
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Artifacts dir: env wins, else first positional arg, else default (gtest
  // flags were already stripped by InitGoogleTest above).
  g_dir = "/tmp/dispatch_order";
  if (argc > 1) {
    g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_DISPATCH_ORDER_DIR")) {
    g_dir = env;
  }

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
