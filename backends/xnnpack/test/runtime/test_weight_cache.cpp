/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstdio>

using namespace ::testing;

using executorch::backends::xnnpack::get_packed_cache_report;
using executorch::backends::xnnpack::packed_cache_path_option_key;
using executorch::backends::xnnpack::PackedCacheHeapReason;
using executorch::backends::xnnpack::save_weight_cache_on_disk_option_key;
using executorch::backends::xnnpack::weight_cache_option_key;
using executorch::backends::xnnpack::workspace_sharing_mode_option_key;
using executorch::backends::xnnpack::WorkspaceSharingMode;
using executorch::backends::xnnpack::xnnpack_backend_key;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;

static void set_and_check_weight_cache_enabled(bool enabled) {
  executorch::runtime::runtime_init();

  BackendOptions<1> backend_options;
  backend_options.set_option(weight_cache_option_key, enabled);

  auto status = executorch::runtime::set_option(
      xnnpack_backend_key, backend_options.view());
  ASSERT_EQ(status, Error::Ok);

  BackendOption read_option;
  strcpy(read_option.key, weight_cache_option_key);
  read_option.value = !enabled;
  status = get_option(xnnpack_backend_key, read_option);

  ASSERT_EQ(std::get<bool>(read_option.value), enabled);
}

static TensorPtr create_input_tensor(float val) {
  std::vector<float> data(1000, val);
  return executorch::extension::make_tensor_ptr({10, 10, 10}, std::move(data));
}

static void load_and_run_model_with_runtime_specs(
    const char* model_path_env,
    const LoadBackendOptionsMap& backend_options_map,
    float input_a,
    float input_b,
    float input_c,
    float expected_output) {
  Module module(std::getenv(model_path_env));

  auto err = module.load(backend_options_map);
  ASSERT_EQ(err, Error::Ok);

  auto a = create_input_tensor(input_a);
  auto b = create_input_tensor(input_b);
  auto c = create_input_tensor(input_c);

  auto result = module.forward({a, b, c});
  ASSERT_TRUE(result.ok());

  auto& output_tensor = result.get()[0].toTensor();
  for (auto i = 0; i < output_tensor.numel(); ++i) {
    ASSERT_EQ(output_tensor.const_data_ptr<float>()[i], expected_output);
  }
}

TEST(WeightCache, SetEnabled) {
  set_and_check_weight_cache_enabled(true);
  set_and_check_weight_cache_enabled(false);
  set_and_check_weight_cache_enabled(true);
}

TEST(WeightCache, SetInvalidType) {
  executorch::runtime::runtime_init();

  // Weight cache option expects a bool, not an int.
  BackendOptions<1> backend_options;
  backend_options.set_option(weight_cache_option_key, 1);

  auto status = executorch::runtime::set_option(
      xnnpack_backend_key, backend_options.view());
  ASSERT_EQ(status, Error::InvalidArgument);
}

TEST(WeightCache, SetMultipleOptions) {
  executorch::runtime::runtime_init();

  // Set both options at once.
  BackendOptions<2> backend_options;
  backend_options.set_option(
      workspace_sharing_mode_option_key,
      static_cast<int>(WorkspaceSharingMode::Global));
  backend_options.set_option(weight_cache_option_key, false);

  auto status = executorch::runtime::set_option(
      xnnpack_backend_key, backend_options.view());
  ASSERT_EQ(status, Error::Ok);

  // Read both back.
  BackendOption read_workspace;
  strcpy(read_workspace.key, workspace_sharing_mode_option_key);
  read_workspace.value = -1;
  status = get_option(xnnpack_backend_key, read_workspace);
  ASSERT_EQ(
      std::get<int>(read_workspace.value),
      static_cast<int>(WorkspaceSharingMode::Global));

  BackendOption read_cache;
  strcpy(read_cache.key, weight_cache_option_key);
  read_cache.value = true;
  status = get_option(xnnpack_backend_key, read_cache);
  ASSERT_EQ(std::get<bool>(read_cache.value), false);
}

TEST(RuntimeSpec, OverridesGlobalWeightCache) {
  executorch::runtime::runtime_init();

  // Set global weight cache to enabled.
  set_and_check_weight_cache_enabled(true);

  // Load a model with runtime spec disabling weight cache.
  BackendOptions<1> xnnpack_opts;
  xnnpack_opts.set_option(weight_cache_option_key, false);
  LoadBackendOptionsMap map;
  map.set_options(xnnpack_backend_key, xnnpack_opts.view());

  // The model should load and run correctly without weight cache.
  load_and_run_model_with_runtime_specs(
      "ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH", map, 1.0, 2.0, 3.0, 9.0);

  // Verify the global setting is still enabled.
  BackendOption read_option;
  strcpy(read_option.key, weight_cache_option_key);
  read_option.value = false;
  get_option(xnnpack_backend_key, read_option);
  ASSERT_EQ(std::get<bool>(read_option.value), true);
}

TEST(PackedCacheStats, GlobalAccessorReachesTheBackendSingleton) {
  executorch::runtime::runtime_init();

  // The wiring under test is get_packed_cache_report() -> the registered
  // backend instance -> XnnpackBackendOptions -> XNNWeightsCacheManager.
  // Hosts call only this entry point, and nothing else in the suite exercises
  // it. Absolute values depend on what else has run in the process, so this
  // asserts reachability and invariants rather than specific counts.
  const auto report = get_packed_cache_report();
  const auto& stats = report.aggregate;

  EXPECT_GE(stats.heap_bytes, 0);
  EXPECT_GE(stats.mapped_bytes, 0);
  EXPECT_GE(stats.file_bytes, 0);
  EXPECT_LT(
      static_cast<int32_t>(stats.heap_reason),
      static_cast<int32_t>(PackedCacheHeapReason::Count));
  EXPECT_NE(stats.heap_reason, PackedCacheHeapReason::NotOptedIn)
      << "NotOptedIn is excluded from heap_bytes and must never be reported";

  // The breakdown must be able to attribute the aggregate: several models
  // share a process, and a single folded number cannot say which one fell
  // back.
  for (const auto& entry : report.per_cache) {
    EXPECT_GE(entry.stats.heap_bytes, 0);
    EXPECT_GE(entry.stats.mapped_bytes, 0);
  }
}

// NOTE: the warm path — load_packed_cache() succeeding and its mapped bytes
// being counted — is deliberately NOT covered here. Producing a loadable
// cache file needs a model with packed weights and a populated index; the
// models wired into this target (ModuleAddLarge / ModuleSubLarge) are
// elementwise and pack nothing, so a cold run writes a zero-entry trailer
// that load_packed_cache correctly rejects. Covering it needs ModuleLinear
// plus its external .ptd and a NamedDataMap, as test_xnn_data_separation
// does. Until then the warm path is verified on device by the
// PackedWeights log line reporting non-zero mapped/cache_file.
