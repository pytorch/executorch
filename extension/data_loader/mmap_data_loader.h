/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {

/**
 * A DataLoader that loads segments from a file, allocating the memory
 * with `malloc()`.
 *
 * Note that this will keep the file open for the duration of its lifetime, to
 * avoid the overhead of opening it again for every load() call.
 */
class MmapDataLoader final : public executorch::runtime::DataLoader {
 public:
  /**
   * Describes how and whether to lock loaded pages with `mlock()`.
   *
   * Using `mlock()` typically loads all of the pages immediately, and will
   * typically ensure that they are not swapped out. The actual behavior
   * will depend on the host system.
   */
  enum class MlockConfig {
    /// Do not call `mlock()` on loaded pages.
    NoMlock,
    /// Call `mlock()` on loaded pages, failing if it fails.
    UseMlock,
    /// Call `mlock()` on loaded pages, ignoring errors if it fails.
    UseMlockIgnoreErrors,
  };

  /**
   * Creates a new MmapDataLoader that wraps the named file. Fails if
   * the file can't be opened for reading or if its size can't be found.
   *
   * @param[in] file_name The path to the file to load from. The file will be
   *     kept open until the MmapDataLoader is destroyed, to avoid the
   *     overhead of opening it again for every load() call.
   * @param[in] mlock_config How and whether to lock loaded pages with
   *     `mlock()`.
   */
  static executorch::runtime::Result<MmapDataLoader> from(
      const char* file_name,
      MlockConfig mlock_config = MlockConfig::UseMlock);

  /// DEPRECATED: Use the lowercase `from()` instead.
  ET_DEPRECATED static executorch::runtime::Result<MmapDataLoader> From(
      const char* file_name,
      MlockConfig mlock_config = MlockConfig::UseMlock) {
    return from(file_name, mlock_config);
  }

  /// DEPRECATED: Use the version of `from()` that takes an MlockConfig.
  ET_DEPRECATED
  static executorch::runtime::Result<MmapDataLoader> From(
      const char* file_name,
      bool use_mlock) {
    MlockConfig mlock_config =
        use_mlock ? MlockConfig::UseMlock : MlockConfig::NoMlock;
    return from(file_name, mlock_config);
  }

  // Movable to be compatible with Result.
  MmapDataLoader(MmapDataLoader&& rhs) noexcept
      : file_name_(rhs.file_name_),
        file_size_(rhs.file_size_),
        page_size_(rhs.page_size_),
        fd_(rhs.fd_),
        mlock_config_(rhs.mlock_config_) {
    const_cast<const char*&>(rhs.file_name_) = nullptr;
    const_cast<size_t&>(rhs.file_size_) = 0;
    const_cast<size_t&>(rhs.page_size_) = 0;
    const_cast<int&>(rhs.fd_) = -1;
    const_cast<MlockConfig&>(rhs.mlock_config_) = MlockConfig::NoMlock;
  }

  ~MmapDataLoader() override;

  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo& segment_info) const override;

  ET_NODISCARD executorch::runtime::Result<size_t> size() const override;

 private:
  MmapDataLoader(
      int fd,
      size_t file_size,
      const char* file_name,
      size_t page_size,
      MlockConfig mlock_config)
      : file_name_(file_name),
        file_size_(file_size),
        page_size_(page_size),
        fd_(fd),
        mlock_config_(mlock_config) {}

  // Not safely copyable.
  MmapDataLoader(const MmapDataLoader&) = delete;
  MmapDataLoader& operator=(const MmapDataLoader&) = delete;
  MmapDataLoader& operator=(MmapDataLoader&&) = delete;

  const char* const file_name_; // String data is owned by the instance.
  const size_t file_size_;
  const size_t page_size_;
  const int fd_; // Owned by the instance.
  const MlockConfig mlock_config_;
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::MmapDataLoader;
} // namespace util
} // namespace executor
} // namespace torch
