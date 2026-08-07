/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUBackend.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/test/native/RequiredDevicePolicy.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  ASSERT_EQ(out_tensor.dim(), 4);
  ASSERT_EQ(out_tensor.size(0), 1);
  ASSERT_EQ(out_tensor.size(1), tc.cmax);
  ASSERT_EQ(out_tensor.size(2), tc.h);
  ASSERT_EQ(out_tensor.size(3), tc.d);
  ASSERT_EQ(static_cast<int>(out_tensor.numel()), cnumel);
  const float* out_data = out_tensor.const_data_ptr<float>();
  EXPECT_EQ(std::memcmp(out_data, ref.data(), ref.size() * sizeof(float)), 0)
      << "update_cache[" << tc.name << "] not bit-exact";
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
    ASSERT_EQ(outputs.size(), 1u) << "bad output count at step " << step;
    ASSERT_TRUE(outputs[0].isTensor()) << "non-tensor output at step " << step;
    const auto& output = outputs[0].toTensor();
    ASSERT_EQ(output.dim(), 4);
    ASSERT_EQ(output.size(0), 1);
    ASSERT_EQ(output.size(1), cmax);
    ASSERT_EQ(output.size(2), rc.h);
    ASSERT_EQ(output.size(3), rc.d);
    ASSERT_EQ(static_cast<int>(output.numel()), cnumel);
    const float* out_data = output.const_data_ptr<float>();

    const int dst_offset = input_pos * rc.h * rc.d;
    for (int i = 0; i < vnumel; i++) {
      if (dst_offset + i < cnumel) {
        ref[dst_offset + i] = value[i];
      }
    }

    EXPECT_EQ(std::memcmp(out_data, ref.data(), ref.size() * sizeof(float)), 0)
        << "step " << step << " (S=" << s << ",pos=" << input_pos << ")";
    std::memcpy(cache.data(), out_data, cache.size() * sizeof(float));
    input_pos += s;
  }
}

constexpr int kDynamicHeads = 2;
constexpr int kDynamicHeadDim = 4;
constexpr int kDynamicMaxCache = 1024;
constexpr int kUpdateCacheWorkgroupSize = 64;

nlohmann::json current_attestation() {
  return nlohmann::json::parse(webgpu_backend_execution_attestation_json());
}

const nlohmann::json* unique_update_cache_command(
    const nlohmann::json& attestation) {
  const auto& commands = attestation.at("canonicalCommands").at("commands");
  const nlohmann::json* match = nullptr;
  for (const auto& command : commands) {
    if (command.at("kind") == "compute" && command.at("enabled") == true &&
        command.at("identity") == "update_cache") {
      EXPECT_EQ(match, nullptr) << "multiple enabled update_cache commands";
      match = &command;
    }
  }
  EXPECT_NE(match, nullptr) << "missing enabled update_cache command";
  return match;
}

void run_dynamic_success(
    Module& module,
    std::vector<float>& full_cache,
    int sequence,
    int capacity,
    int input_pos,
    int salt,
    uint64_t* last_ordinal) {
  const int stride = kDynamicHeads * kDynamicHeadDim;
  const int value_numel = sequence * stride;
  const int cache_numel = capacity * stride;
  std::vector<float> value(static_cast<size_t>(value_numel));
  for (int i = 0; i < value_numel; ++i) {
    value[i] = static_cast<float>(salt * 100000 + i);
  }
  std::vector<float> cache(
      full_cache.begin(), full_cache.begin() + cache_numel);
  std::vector<float> expected(cache);
  std::memcpy(
      expected.data() + static_cast<size_t>(input_pos) * stride,
      value.data(),
      value.size() * sizeof(float));

  auto value_tensor = make_tensor_ptr(
      {1, sequence, kDynamicHeads, kDynamicHeadDim}, std::move(value));
  auto cache_tensor = make_tensor_ptr(
      {1, capacity, kDynamicHeads, kDynamicHeadDim}, std::move(cache));
  auto position_tensor =
      make_tensor_ptr({1}, std::vector<int64_t>{input_pos});
  auto result = module.forward(
      {EValue(value_tensor), EValue(cache_tensor), EValue(position_tensor)});
  ASSERT_TRUE(result.ok()) << "S=" << sequence << " C=" << capacity
                           << " pos=" << input_pos
                           << " error=" << static_cast<int>(result.error());
  size_t cache_output_count = 0;
  for (const auto& output_value : result.get()) {
    if (!output_value.isTensor()) {
      continue;
    }
    const auto& output = output_value.toTensor();
    if (output.dim() != 4 || output.size(0) != 1 ||
        output.size(1) != capacity || output.size(2) != kDynamicHeads ||
        output.size(3) != kDynamicHeadDim) {
      continue;
    }
    ++cache_output_count;
    ASSERT_EQ(static_cast<int>(output.numel()), cache_numel);
    const float* output_data = output.const_data_ptr<float>();
    ASSERT_EQ(
        std::memcmp(
            output_data, expected.data(), expected.size() * sizeof(float)),
        0);
    std::memcpy(
        full_cache.data(), output_data, expected.size() * sizeof(float));
  }
  ASSERT_EQ(cache_output_count, 1u);

  const nlohmann::json attestation = current_attestation();
  ASSERT_TRUE(attestation.at("completed").get<bool>());
  const uint64_t ordinal = attestation.at("executionOrdinal").get<uint64_t>();
  if (*last_ordinal != 0) {
    EXPECT_EQ(ordinal, *last_ordinal + 1);
  }
  *last_ordinal = ordinal;
  const auto* command = unique_update_cache_command(attestation);
  ASSERT_NE(command, nullptr);
  const uint32_t expected_grid = static_cast<uint32_t>(
      (value_numel + kUpdateCacheWorkgroupSize - 1) /
      kUpdateCacheWorkgroupSize);
  EXPECT_EQ(
      command->at("grid"),
      nlohmann::json::array({expected_grid, 1u, 1u}));
}

void expect_dynamic_failure(
    Module& module,
    const std::vector<float>& full_cache,
    int sequence,
    int capacity,
    int input_pos) {
  const int stride = kDynamicHeads * kDynamicHeadDim;
  const std::string before = webgpu_backend_execution_attestation_json();
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::vector<float> value(
        static_cast<size_t>(sequence * stride), 7.0f);
    std::vector<float> cache(
        full_cache.begin(), full_cache.begin() + capacity * stride);
    const std::vector<float> expected_value(value);
    const std::vector<float> expected_cache(cache);
    auto value_tensor = make_tensor_ptr(
        {1, sequence, kDynamicHeads, kDynamicHeadDim}, std::move(value));
    auto cache_tensor = make_tensor_ptr(
        {1, capacity, kDynamicHeads, kDynamicHeadDim}, std::move(cache));
    auto position_tensor =
        make_tensor_ptr({1}, std::vector<int64_t>{input_pos});

    auto result = module.forward(
        {EValue(value_tensor), EValue(cache_tensor), EValue(position_tensor)});

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error(), Error::Internal);
    EXPECT_EQ(
        std::memcmp(
            value_tensor->const_data_ptr<float>(),
            expected_value.data(),
            expected_value.size() * sizeof(float)),
        0);
    EXPECT_EQ(
        std::memcmp(
            cache_tensor->const_data_ptr<float>(),
            expected_cache.data(),
            expected_cache.size() * sizeof(float)),
        0);
    EXPECT_EQ(webgpu_backend_execution_attestation_json(), before);
  }
}

void run_intermediate_success(
    Module& module,
    std::vector<float>& full_cache,
    int sequence,
    int input_pos,
    int salt,
    uint64_t* last_ordinal) {
  constexpr int kCapacity = 768;
  const int stride = kDynamicHeads * kDynamicHeadDim;
  const int value_numel = sequence * stride;
  const int cache_numel = kCapacity * stride;
  std::vector<float> value(static_cast<size_t>(value_numel));
  std::vector<float> transformed(static_cast<size_t>(value_numel));
  for (int i = 0; i < value_numel; ++i) {
    value[i] = static_cast<float>((i + salt) % 33 - 16) * 0.125f;
    transformed[i] = 1.0f / (1.0f + std::exp(-value[i]));
  }
  std::vector<float> cache(full_cache.begin(), full_cache.begin() + cache_numel);
  const std::vector<float> before(cache);

  auto value_tensor = make_tensor_ptr(
      {1, sequence, kDynamicHeads, kDynamicHeadDim}, std::move(value));
  auto cache_tensor = make_tensor_ptr(
      {1, kCapacity, kDynamicHeads, kDynamicHeadDim}, std::move(cache));
  auto position_tensor =
      make_tensor_ptr({1}, std::vector<int64_t>{input_pos});
  auto result = module.forward(
      {EValue(value_tensor), EValue(cache_tensor), EValue(position_tensor)});
  ASSERT_TRUE(result.ok()) << "S=" << sequence << " pos=" << input_pos
                           << " error=" << static_cast<int>(result.error());

  size_t cache_output_count = 0;
  for (const auto& output_value : result.get()) {
    if (!output_value.isTensor()) {
      continue;
    }
    const auto& output = output_value.toTensor();
    if (output.dim() != 4 || output.size(0) != 1 ||
        output.size(1) != kCapacity || output.size(2) != kDynamicHeads ||
        output.size(3) != kDynamicHeadDim) {
      continue;
    }
    ++cache_output_count;
    const float* output_data = output.const_data_ptr<float>();
    const size_t write_begin = static_cast<size_t>(input_pos) * stride;
    const size_t write_end = write_begin + transformed.size();
    for (size_t i = 0; i < static_cast<size_t>(cache_numel); ++i) {
      if (i >= write_begin && i < write_end) {
        EXPECT_NEAR(output_data[i], transformed[i - write_begin], 1.0e-3f);
      } else {
        EXPECT_EQ(
            std::memcmp(output_data + i, before.data() + i, sizeof(float)),
            0);
      }
    }
    std::memcpy(
        full_cache.data(),
        output_data,
        static_cast<size_t>(cache_numel) * sizeof(float));
  }
  ASSERT_EQ(cache_output_count, 1u);

  const nlohmann::json attestation = current_attestation();
  ASSERT_TRUE(attestation.at("completed").get<bool>());
  const uint64_t ordinal = attestation.at("executionOrdinal").get<uint64_t>();
  if (*last_ordinal != 0) {
    EXPECT_EQ(ordinal, *last_ordinal + 1);
  }
  *last_ordinal = ordinal;
  const auto* command = unique_update_cache_command(attestation);
  ASSERT_NE(command, nullptr);
  const uint32_t expected_grid = static_cast<uint32_t>(
      (value_numel + kUpdateCacheWorkgroupSize - 1) /
      kUpdateCacheWorkgroupSize);
  EXPECT_EQ(
      command->at("grid"),
      nlohmann::json::array({expected_grid, 1u, 1u}));
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

TEST(UpdateCache, DynamicSymIntShapeCapacityBoundsAndRecovery) {
  Module module(g_dir + "/dynamic.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load dynamic.pte";
  std::vector<float> full_cache(
      static_cast<size_t>(kDynamicMaxCache) * kDynamicHeads * kDynamicHeadDim);
  for (size_t i = 0; i < full_cache.size(); ++i) {
    full_cache[i] = static_cast<float>(i) + 0.25f;
  }
  uint64_t ordinal = 0;

  run_dynamic_success(module, full_cache, 512, 1024, 0, 1, &ordinal);
  run_dynamic_success(module, full_cache, 1, 1024, 0, 2, &ordinal);
  run_dynamic_success(module, full_cache, 1, 768, 0, 3, &ordinal);
  run_dynamic_success(module, full_cache, 1, 768, 512, 4, &ordinal);
  expect_dynamic_failure(module, full_cache, 1, 512, 512);
  run_dynamic_success(module, full_cache, 1, 768, 513, 5, &ordinal);
  run_dynamic_success(module, full_cache, 512, 1024, 0, 6, &ordinal);
  run_dynamic_success(module, full_cache, 1, 1024, 0, 7, &ordinal);

  expect_dynamic_failure(module, full_cache, 1, 1024, -1);
  run_dynamic_success(module, full_cache, 1, 1024, 1, 8, &ordinal);
  expect_dynamic_failure(module, full_cache, 512, 1024, 513);
  run_dynamic_success(module, full_cache, 1, 1024, 2, 9, &ordinal);
}

TEST(UpdateCache, DynamicIntermediateValueRefreshesAfterTensorFixpoint) {
  Module module(g_dir + "/dynamic_intermediate.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load dynamic_intermediate.pte";
  std::vector<float> full_cache(
      static_cast<size_t>(768) * kDynamicHeads * kDynamicHeadDim);
  for (size_t i = 0; i < full_cache.size(); ++i) {
    full_cache[i] = static_cast<float>(i) + 0.75f;
  }
  uint64_t ordinal = 0;

  run_intermediate_success(module, full_cache, 512, 0, 1, &ordinal);
  run_intermediate_success(module, full_cache, 1, 0, 2, &ordinal);
  run_intermediate_success(module, full_cache, 512, 0, 3, &ordinal);
  run_intermediate_success(module, full_cache, 1, 512, 4, &ordinal);
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
    std::printf("SKIP: no WebGPU device (%s)\n", e.what());
    return required_device_failure_exit_code(
        std::getenv("WEBGPU_REQUIRE_DEVICE") != nullptr);
  }
  set_default_webgpu_context(&ctx);
  std::printf("WebGPU device acquired (native)\n");

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
