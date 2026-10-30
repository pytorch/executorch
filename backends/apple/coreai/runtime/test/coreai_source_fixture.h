/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <map>
#include <string>
#include "coreai_bookmark_fixture.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_manifest_fixture.h"
#include "coreai_source_bundle.h"

namespace executorch::backends::coreai::testing {

class TestData final : public runtime::NamedDataMap {
 public:
  mutable std::atomic<int> releases{0};
  mutable std::atomic<int> requests{0};
  mutable std::atomic<int> attempts{0};
  mutable std::atomic<int> metadata_requests{0};
  int fail_at = 0;
  runtime::Error failure_error = runtime::Error::AccessFailed;
  std::string required_prefix;
  std::map<std::string, std::string> files{
      {"coreai/ab/model.aimodel/graph.bin", std::string("model data", 11)}};
  runtime::Result<const runtime::TensorLayout> get_tensor_layout(
      std::string_view) const override;
  runtime::Result<runtime::FreeableBuffer> get_data(
      std::string_view key) const override;
  runtime::Error load_data_into(std::string_view, void*, size_t) const override;
  runtime::Result<uint32_t> get_num_keys() const override;
  runtime::Result<const char*> get_key(uint32_t i) const override;
};

void aot_data(TestData& data, NSString* arch);
runtime::Result<SelectedBundle> source_manifest(bool aot = false);
NSString* source_key(const SelectedBundle& bundle);
NSURL* staging_root(NSURL* cache_root);
NSURL* source_entry(NSURL* cache_root, const SelectedBundle& bundle);

class CoreAISourceTest : public ::testing::Test {
 protected:
  void check_concurrent_source();
  runtime::Result<NSURL*> prepare_source_bundle(
      const SelectedBundle& bundle,
      const runtime::NamedDataMap* data,
      NSString* cache_root,
      CorruptBundlePolicy corrupt_policy = CorruptBundlePolicy::Reject);
};

} // namespace executorch::backends::coreai::testing
