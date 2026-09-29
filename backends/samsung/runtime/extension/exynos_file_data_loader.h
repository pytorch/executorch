/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace backends {
namespace enn {

/**
 * Loads a file from disk into buffers backed by ENN shared memory (dmabuf), so
 * that segments handed to the ENN backend can be opened without an extra copy.
 *
 * Mirrors executorch::extension::FileDataLoader, which cannot be reused
 * directly because it is final and always allocates from the heap.
 */
class ExynosFileDataLoader final : public executorch::runtime::DataLoader {
 public:
  // `alignment` must not exceed the system page size: ENN shared memory
  // buffers are only guaranteed to be page-aligned, and larger requests are
  // rejected rather than silently under-aligned.
  static executorch::runtime::Result<ExynosFileDataLoader> from(
      const char* file_name,
      size_t alignment = alignof(std::max_align_t));

  ExynosFileDataLoader(ExynosFileDataLoader&& rhs) noexcept
      : file_name_(rhs.file_name_),
        file_size_(rhs.file_size_),
        alignment_(rhs.alignment_),
        fd_(rhs.fd_) {
    const_cast<const char*&>(rhs.file_name_) = nullptr;
    const_cast<size_t&>(rhs.file_size_) = 0;
    const_cast<std::align_val_t&>(rhs.alignment_) = {};
    const_cast<int&>(rhs.fd_) = -1;
  }

  ~ExynosFileDataLoader() override;

  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo& segment_info) const override;

  ET_NODISCARD executorch::runtime::Result<size_t> size() const override;

  ET_NODISCARD executorch::runtime::Error load_into(
      size_t offset,
      size_t size,
      ET_UNUSED const SegmentInfo& segment_info,
      void* buffer) const override;

 private:
  ExynosFileDataLoader(
      int fd,
      size_t file_size,
      size_t alignment,
      const char* file_name)
      : file_name_(file_name),
        file_size_(file_size),
        alignment_{alignment},
        fd_(fd) {}

  // Not safely copyable.
  ExynosFileDataLoader(const ExynosFileDataLoader&) = delete;
  ExynosFileDataLoader& operator=(const ExynosFileDataLoader&) = delete;
  ExynosFileDataLoader& operator=(ExynosFileDataLoader&&) = delete;

  const char* const file_name_; // Owned by the instance.
  const size_t file_size_;
  const std::align_val_t alignment_;
  const int fd_; // Owned by the instance.
};

} // namespace enn
} // namespace backends
} // namespace executorch
