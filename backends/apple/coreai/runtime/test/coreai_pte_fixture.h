/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/schema/program_generated.h>
#include <gtest/gtest.h>

#include <atomic>
#include <optional>
#include <string>
#include <vector>

namespace executorch::backends::coreai::testing {

struct CacheDelegateSpec {
  const char* backend;
  NSData* data;
  bool segmented = false;
  int index_override = -1;
  bool missing_processed = false;
  std::optional<size_t> backend_length = std::nullopt;
};

struct CacheMethodSpec {
  const char* name;
  std::vector<CacheDelegateSpec> delegates;
};

class FBSyntheticPTE final : public executorch::runtime::DataLoader {
 public:
  enum class LoadFault { None, Error };
  struct Request {
    size_t offset;
    size_t size;
    SegmentInfo::Type type;
    size_t index;
    std::string descriptor;
    const void* data = nullptr;
  };

  std::vector<uint8_t> bytes;
  mutable std::vector<Request> requests;
  mutable std::vector<SegmentInfo::Type> release_order;
  mutable bool program_alive_at_backend_release = true;
  executorch::runtime::Error size_error = executorch::runtime::Error::Ok;
  executorch::runtime::Error load_error = executorch::runtime::Error::NotSupported;
  int fault_call = -1;
  LoadFault load_fault = LoadFault::None;
  mutable std::atomic<int> program_loads{0};
  mutable std::atomic<int> program_releases{0};
  mutable std::atomic<int> backend_loads{0};
  mutable std::atomic<int> backend_releases{0};
  mutable std::vector<size_t> backend_indices;
  mutable std::vector<std::string> backend_descriptors;
  int fail_backend_index = -1;

  explicit FBSyntheticPTE(const std::vector<CacheMethodSpec>& methods, bool extended_header = true);

  executorch::runtime::Result<size_t> size() const override;
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset, size_t size, const SegmentInfo& info) const override;
  ::testing::AssertionResult check_released() const;
};

}  // namespace executorch::backends::coreai::testing
