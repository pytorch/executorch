/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUBackend.h>
#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

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
  struct Mode {
    bool single_compute_pass;
    size_t cap;
  };
  const std::vector<Mode> modes = {
      {false, 0}, {true, 0}, {true, 2}, {false, 0}};
  std::string command_inventory;
  uint64_t prior_ordinal = 0;
  for (const Mode mode : modes) {
    auto x = make_tensor_ptr(sizes, std::vector<float>(input));
    WebGPUExecutionOptions options;
    options.single_compute_pass = mode.single_compute_pass;
    options.max_compute_dispatches_per_pass = mode.cap;
    auto result = with_webgpu_execution_options(
        options, [&]() { return module.forward({EValue(x)}); });
    ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                             << ")";
    const auto& outputs = result.get();
    ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor())
        << "no tensor output";
    const auto& out_tensor = outputs[0].toTensor();
    ASSERT_EQ(static_cast<size_t>(out_tensor.numel()), golden.size());
    const float* out_data = out_tensor.const_data_ptr<float>();

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < golden.size(); i++) {
      const float abs_err = std::abs(out_data[i] - golden[i]);
      max_abs_err = std::max(max_abs_err, abs_err);
      const float denom = std::max(std::abs(golden[i]), 1e-6f);
      max_rel_err = std::max(max_rel_err, abs_err / denom);
    }
    EXPECT_FALSE(max_abs_err > 1e-3f && max_rel_err > 1e-3f)
        << "dispatch_order[" << name
        << "] exceeds tolerance 1e-3 (max_abs_err=" << max_abs_err
        << " max_rel_err=" << max_rel_err << ")";

    const auto attestation = nlohmann::json::parse(
        webgpu_backend_execution_attestation_json());
    EXPECT_EQ(
        attestation.at("requested").get<bool>(), mode.single_compute_pass);
    EXPECT_EQ(
        attestation.at("applied").get<bool>(), mode.single_compute_pass);
    EXPECT_TRUE(attestation.at("completed").get<bool>());
    EXPECT_EQ(attestation.at("queueSubmitCount").get<size_t>(), 1u);
    const size_t active =
        attestation.at("activeComputeCount").get<size_t>();
    const size_t runs =
        attestation.at("maximalComputeRuns").get<size_t>();
    ASSERT_GT(active, 0u);
    ASSERT_GT(runs, 0u);
    size_t expected_passes = 0;
    size_t dispatches_in_pass = 0;
    for (const auto& command :
         attestation.at("canonicalCommands").at("commands")) {
      if (command.at("kind") != "compute") {
        if (command.at("enabled").get<bool>()) {
          dispatches_in_pass = 0;
        }
        continue;
      }
      if (!command.at("enabled").get<bool>() ||
          command.at("zeroGrid").get<bool>()) {
        continue;
      }
      if (!mode.single_compute_pass || dispatches_in_pass == 0) {
        ++expected_passes;
      }
      ++dispatches_in_pass;
      if (!mode.single_compute_pass ||
          (mode.cap != 0 && dispatches_in_pass >= mode.cap)) {
        dispatches_in_pass = 0;
      }
    }
    EXPECT_EQ(
        attestation.at("encodedComputePasses").get<size_t>(), expected_passes);
    EXPECT_EQ(
        attestation.at("maxComputeDispatchesPerPass").get<size_t>(), mode.cap);
    const uint64_t ordinal =
        attestation.at("executionOrdinal").get<uint64_t>();
    EXPECT_EQ(ordinal, prior_ordinal + 1);
    prior_ordinal = ordinal;
    const std::string current_inventory =
        attestation.at("canonicalCommands").dump();
    if (command_inventory.empty()) {
      command_inventory = current_inventory;
    } else {
      EXPECT_EQ(current_inventory, command_inventory);
    }
  }
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
