/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <executorch/runtime/backend/backend_cache.h>

namespace executorch {
namespace extension {

/**
 * Filesystem-based BackendCache implementation.
 *
 * Keys map to file paths under a cache directory:
 *   {cache_dir}/{backend_id}/{delegate_index}/{key}
 *
 * Uses atomic writes (write to temp + rename) to prevent corruption.
 *
 * Concurrency: safe for concurrent reads from multiple threads/processes.
 * Concurrent writes use last-writer-wins semantics via atomic rename.
 */
class FileBackendCache final : public runtime::BackendCache {
 public:
  explicit FileBackendCache(std::string cache_dir);

  runtime::Result<runtime::FreeableBuffer> load(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      size_t alignment = alignof(std::max_align_t)) const override;

  runtime::Error save(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      const void* data,
      size_t size) override;

  runtime::Error remove(
      const char* backend_id,
      size_t delegate_index,
      const char* key) override;

 private:
  std::string cache_dir_;

  std::string key_to_path(
      const char* backend_id,
      size_t delegate_index,
      const char* key) const;
};

} // namespace extension
} // namespace executorch
