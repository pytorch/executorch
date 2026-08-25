/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/backends/cuda/runtime/cuda_weight_manifest.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/named_data_map.h>

namespace executorch::backends::cuda {

class CudaWeightCache final {
 public:
  runtime::Error load(
      CudaDelegateHandle* handle,
      const runtime::NamedDataMap* named_data_map,
      const CudaFqnWeightManifest& manifest) const;

 private:
  static runtime::Error validate_view(const CudaFqnWeightEntry& entry);

  runtime::Error acquire_storage(
      const runtime::NamedDataMap* named_data_map,
      const CudaFqnWeightEntry& entry,
      uintptr_t logical_scope,
      int device_index,
      std::shared_ptr<CudaWeightStorage>& storage,
      bool& reused) const;

  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, std::weak_ptr<CudaWeightStorage>>
      storages_;
};

} // namespace executorch::backends::cuda
