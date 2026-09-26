/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/named_data_map.h>

namespace executorch::backends::cuda {

class CudaWeightCache final {
 public:
  static constexpr char kFormatMagic[] = "ETCUDAFQN0";
  static constexpr size_t kFormatMagicSize = sizeof(kFormatMagic) - 1;

  struct Variant {
    uint32_t target_sm{0};
    uint32_t ptx_compute{0};
    std::string so_blob_key;
    bool fallback_only{false};
  };

  struct Entry {
    std::string fqn;
    std::string storage_key;
    uint64_t storage_nbytes{0};
    int32_t dtype{0};
    int32_t device_type{0};
    int64_t storage_offset{0};
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
  };

  struct Metadata {
    std::vector<Variant> variants;
    std::vector<Entry> entries;
  };

  static bool is_serialized(const void* data, size_t size);

  static runtime::Error
  parse(const void* data, size_t size, Metadata& metadata);

  static runtime::Error select_variant(
      const Metadata& metadata,
      uint32_t current_sm,
      size_t& variant_index,
      bool& uses_ptx_fallback);

  runtime::Error load(
      CudaDelegateHandle* handle,
      const runtime::NamedDataMap* named_data_map,
      const Metadata& metadata) const;

 private:
  static runtime::Error validate_view(const Entry& entry);

  runtime::Error acquire_storage(
      const runtime::NamedDataMap* named_data_map,
      const Entry& entry,
      uintptr_t logical_scope,
      int device_index,
      std::shared_ptr<CudaWeightStorage>& storage,
      bool& reused) const;

  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, std::weak_ptr<CudaWeightStorage>>
      storages_;
};

} // namespace executorch::backends::cuda
