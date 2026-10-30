/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * ESP-IDF replacement for extension/data_loader/mmap_data_loader.cpp.
 *
 * ESP-IDF has no mmap (no sys/mman.h, no virtual memory), so the
 * upstream implementation cannot build. extension::Module references
 * MmapDataLoader for its file-path Mmap load modes, so the class must
 * exist to link; every entry point here fails cleanly with
 * Error::NotSupported. BufferDataLoader (flash XIP) and
 * FileDataLoader (VFS) remain fully functional.
 *
 * The generator (build_idf_component.sh) copies this file in place of
 * the upstream mmap_data_loader.cpp and records the substitution in
 * the component's VERSION file.
 */

#include <executorch/extension/data_loader/mmap_data_loader.h>

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {

using runtime::Error;
using runtime::FreeableBuffer;
using runtime::Result;

Result<MmapDataLoader> MmapDataLoader::from(
    ET_UNUSED const char* file_name,
    ET_UNUSED MlockConfig mlock_config) {
  ET_LOG(
      Error,
      "MmapDataLoader is not supported on ESP-IDF (no mmap); "
      "use BufferDataLoader or FileDataLoader");
  return Error::NotSupported;
}

MmapDataLoader::~MmapDataLoader() {}

Result<FreeableBuffer> MmapDataLoader::load(
    ET_UNUSED size_t offset,
    ET_UNUSED size_t size,
    ET_UNUSED const DataLoader::SegmentInfo& segment_info) const {
  return Error::NotSupported;
}

Result<size_t> MmapDataLoader::size() const {
  return Error::NotSupported;
}

Error MmapDataLoader::load_into(
    ET_UNUSED size_t offset,
    ET_UNUSED size_t size,
    ET_UNUSED const SegmentInfo& segment_info,
    ET_UNUSED void* buffer) const {
  return Error::NotSupported;
}

Error MmapDataLoader::validate_input(
    ET_UNUSED size_t offset,
    ET_UNUSED size_t size) const {
  return Error::NotSupported;
}

} // namespace extension
} // namespace executorch
