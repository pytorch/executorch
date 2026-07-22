/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>
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

constexpr int kWidth = 64;
constexpr float kPoison = -12345.0f;

std::string g_dir; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

std::vector<float> read_bin(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "read_bin: cannot open %s\n", path.c_str());
    return {};
  }
  const std::streamsize bytes = file.tellg();
  if (bytes < 0 || bytes % sizeof(float) != 0) {
    fprintf(
        stderr,
        "read_bin: %s has non-float size %lld\n",
        path.c_str(),
        static_cast<long long>(bytes));
    return {};
  }
  file.seekg(0);
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  file.read(reinterpret_cast<char*>(values.data()), bytes);
  return values;
}

void expect_matches(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); i++) {
    const float abs_err = std::fabs(actual[i] - expected[i]);
    const float denom = std::max(std::fabs(expected[i]), 1e-12f);
    const float rel_err = abs_err / denom;
    EXPECT_TRUE(abs_err <= 1e-3f || rel_err <= 1e-3f)
        << "index=" << i << " actual=" << actual[i]
        << " expected=" << expected[i] << " abs=" << abs_err
        << " rel=" << rel_err;
  }
}

std::vector<float> input_values() {
  auto input = read_bin(g_dir + "/input.bin");
  EXPECT_EQ(input.size(), static_cast<size_t>(kWidth));
  return input;
}

Result<std::vector<EValue>> forward_with_suppression(
    Module& module,
    const TensorPtr& input,
    const void* discardable_output_data) {
  return with_webgpu_execution_options(
      WebGPUExecutionOptions{discardable_output_data, true},
      [&]() { return module.forward({EValue(input)}); });
}

TEST(OutputSuppression, DirectFinalQ4SkipsOnlyDiscardedInvocation) {
  Module module(g_dir + "/direct_final_q4.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok);

  auto input = make_tensor_ptr({1, kWidth}, input_values());
  std::vector<float> suppressed(kWidth, kPoison);
  auto suppressed_tensor =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(suppressed.data()));
  ASSERT_EQ(module.set_output(EValue(suppressed_tensor)), Error::Ok);
  const auto result =
      forward_with_suppression(module, input, suppressed.data());
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(std::all_of(suppressed.begin(), suppressed.end(), [](float v) {
    return v == kPoison;
  }));

  std::vector<float> terminal(kWidth, kPoison);
  auto terminal_tensor =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(terminal.data()));
  ASSERT_EQ(module.set_output(EValue(terminal_tensor)), Error::Ok);
  const auto terminal_result = module.forward({EValue(input)});
  ASSERT_TRUE(terminal_result.ok());
  expect_matches(
      terminal, read_bin(g_dir + "/direct_final_q4.output0.golden.bin"));
}

TEST(OutputSuppression, Q4FeedingLaterAddIsNotSuppressible) {
  Module module(g_dir + "/q4_then_add.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok);

  auto input = make_tensor_ptr({1, kWidth}, input_values());
  std::vector<float> output0(kWidth, kPoison);
  std::vector<float> output1(kWidth, kPoison);
  auto tensor0 =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(output0.data()));
  auto tensor1 =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(output1.data()));
  ASSERT_EQ(module.set_outputs({EValue(tensor0), EValue(tensor1)}), Error::Ok);
  const auto result = forward_with_suppression(module, input, output0.data());
  ASSERT_TRUE(result.ok());
  expect_matches(output0, read_bin(g_dir + "/q4_then_add.output0.golden.bin"));
  expect_matches(output1, read_bin(g_dir + "/q4_then_add.output1.golden.bin"));
}

TEST(OutputSuppression, UnrelatedOutputStillCopies) {
  Module module(g_dir + "/unrelated_then_final_q4.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok);

  auto input = make_tensor_ptr({1, kWidth}, input_values());
  std::vector<float> unrelated(kWidth, kPoison);
  std::vector<float> suppressed(kWidth, kPoison);
  auto unrelated_tensor =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(unrelated.data()));
  auto suppressed_tensor =
      make_tensor_ptr({1, kWidth}, static_cast<void*>(suppressed.data()));
  ASSERT_EQ(
      module.set_outputs({EValue(unrelated_tensor), EValue(suppressed_tensor)}),
      Error::Ok);
  const auto result =
      forward_with_suppression(module, input, suppressed.data());
  ASSERT_TRUE(result.ok());
  expect_matches(
      unrelated,
      read_bin(g_dir + "/unrelated_then_final_q4.output0.golden.bin"));
  EXPECT_TRUE(std::all_of(suppressed.begin(), suppressed.end(), [](float v) {
    return v == kPoison;
  }));
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  g_dir = "/tmp/output_suppression";
  if (argc > 1) {
    g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_OUTPUT_SUPPRESSION_DIR")) {
    g_dir = env;
  }

  WebGPUContext context;
  try {
    context = create_webgpu_context();
  } catch (const std::exception& error) {
    std::printf("SKIP: no WebGPU device (%s)\n", error.what());
    return 0;
  }
  set_default_webgpu_context(&context);
  const int result = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(context);
  return result;
}
