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
#include <cstdint>
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
  const size_t bytes =
      static_cast<size_t>(f.tellg()) / sizeof(float) * sizeof(float);
  f.seekg(0);
  std::vector<float> data(bytes / sizeof(float));
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(bytes));
  return data;
}

std::vector<int32_t> read_i32_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const size_t bytes =
      static_cast<size_t>(f.tellg()) / sizeof(int32_t) * sizeof(int32_t);
  f.seekg(0);
  std::vector<int32_t> data(bytes / sizeof(int32_t));
  f.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(bytes));
  return data;
}

// index.Tensor: self [n] float, idx [m] int64 -> output [m]. Names mirror
// test_index.py CONFIGS (self/idx/golden bins written per case).
void run_case(const char* name) {
  const std::string base = g_dir + "/" + name;
  std::vector<float> self_data = read_f32_bin(base + ".self.bin");
  std::vector<int32_t> idx32 = read_i32_bin(base + ".idx.bin");
  std::vector<float> golden = read_f32_bin(base + ".golden.bin");
  ASSERT_FALSE(self_data.empty() || idx32.empty() || golden.empty())
      << "could not read self/idx/golden for " << name;

  Module module(base + ".pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load " << name << ".pte";

  const int32_t n = static_cast<int32_t>(self_data.size());
  const int32_t m = static_cast<int32_t>(idx32.size());
  auto x = make_tensor_ptr({n}, std::vector<float>(self_data));
  // int64 at the program boundary; copy_inputs narrows to the int32 buffer.
  std::vector<int64_t> idx64(idx32.begin(), idx32.end());
  auto idx = make_tensor_ptr({m}, std::vector<int64_t>(idx64));

  auto result = module.forward({EValue(x), EValue(idx)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";

  const auto& outputs = result.get();
  // index.Tensor has exactly one output of shape [num_indices]; fail loud else.
  ASSERT_TRUE(outputs.size() == 1 && outputs[0].isTensor())
      << "expected exactly one tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_TRUE(out_tensor.dim() == 1 && out_tensor.size(0) == m)
      << "output shape mismatch (dim " << (int)out_tensor.dim() << " size0 "
      << (int)(out_tensor.dim() == 1 ? out_tensor.size(0) : -1)
      << ", expected [" << m << "])";
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
  EXPECT_LE(max_abs_err, 1e-3f) << name << " max_abs_err=" << max_abs_err
                                << " (" << golden.size() << " elements)";
  EXPECT_LE(max_rel_err, 1e-3f) << name << " max_rel_err=" << max_rel_err
                                << " (" << golden.size() << " elements)";
}

} // namespace

TEST(Index, N16M5) {
  run_case("index_n16_m5");
}

TEST(Index, N8Rev) {
  run_case("index_n8_rev");
}

TEST(Index, N32M3) {
  run_case("index_n32_m3");
}

TEST(Index, N4Rep) {
  run_case("index_n4_rep");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Artifacts dir: env wins, else first positional arg, else default (gtest
  // flags were already stripped by InitGoogleTest above).
  g_dir = "/tmp/index";
  if (argc > 1) {
    g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_INDEX_DIR")) {
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
