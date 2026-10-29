/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_source_fixture.h"

#include <iterator>
#include "coreai_bookmarks.h"
#include "coreai_cache_key.h"

namespace executorch::backends::coreai::testing {
using runtime::Error;
using runtime::Result;

Result<const runtime::TensorLayout> TestData::get_tensor_layout(
    std::string_view) const {
  return Error::NotSupported;
}

Result<runtime::FreeableBuffer> TestData::get_data(std::string_view key) const {
  if (!required_prefix.empty() &&
      key.substr(0, required_prefix.size()) != required_prefix) {
    ADD_FAILURE() << "NamedDataMap key " << key
                  << " is outside selected prefix " << required_prefix;
    return Error::InvalidArgument;
  }
  if (++attempts == fail_at) {
    return failure_error;
  }
  auto found = files.find(std::string(key));
  if (found == files.end()) {
    return Error::NotFound;
  }
  ++requests;
  return runtime::FreeableBuffer(
      found->second.data(),
      found->second.size(),
      [](void* context, void*, size_t) {
        ++static_cast<TestData*>(context)->releases;
      },
      const_cast<TestData*>(this));
}

Error TestData::load_data_into(std::string_view, void*, size_t) const {
  return Error::NotSupported;
}

Result<uint32_t> TestData::get_num_keys() const {
  ++metadata_requests;
  return static_cast<uint32_t>(files.size());
}

Result<const char*> TestData::get_key(uint32_t i) const {
  ++metadata_requests;
  if (i >= files.size()) {
    return Error::NotFound;
  }
  auto found = files.begin();
  std::advance(found, i);
  return found->first.c_str();
}

void aot_data(TestData& data, NSString* arch) {
  data.required_prefix =
      [NSString stringWithFormat:@"coreai/ab/model.%@.aimodelc/", arch]
          .UTF8String;
  data.files.clear();
  data.files[data.required_prefix + "graph.bin"] =
      std::string("compiled\0graph", 14);
  data.files[data.required_prefix + "nested/weights.bin"] = "weights";
}

Result<NSURL*> CoreAISourceTest::prepare_source_bundle(
    const SelectedBundle& bundle,
    const runtime::NamedDataMap* data,
    NSString* cache_root,
    CorruptBundlePolicy corrupt_policy) {
  auto key = cache_variant_key(bundle, @"macOS", @"arch_b", @"default");
  if (!key.ok()) {
    ADD_FAILURE() << "Cannot compute source fixture key";
    return key.error();
  }
  auto lock = lock_cache_entry(cache_root, key.get());
  if (!lock.ok()) {
    ADD_FAILURE() << "Cannot acquire source fixture lock";
    return lock.error();
  }
  return executorch::backends::coreai::prepare_source_bundle(
      bundle, data, *lock.get(), corrupt_policy);
}

} // namespace executorch::backends::coreai::testing
