/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>
#include <cstring>

namespace executorch {
namespace extension {

/**
 * A DataLoader that wraps a pre-allocated buffer. The FreeableBuffers
 * that it returns do not actually free any data.
 *
 * This can be used to wrap data that is directly embedded into the firmware
 * image, or to wrap data that was allocated elsewhere.
 */
class BufferDataLoader final : public executorch::runtime::DataLoader {
 public:
  BufferDataLoader(const void* data, size_t size)
      : data_(reinterpret_cast<const uint8_t*>(data)), size_(size) {}

  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      ET_UNUSED const DataLoader::SegmentInfo& segment_info) const override {
    ET_CHECK_OR_RETURN_ERROR(
        offset + size <= size_,
        InvalidArgument,
        "offset %zu + size %zu > size_ %zu",
        offset,
        size,
        size_);
    return executorch::runtime::FreeableBuffer(
        data_ + offset, size, /*free_fn=*/nullptr);
  }

  ET_NODISCARD executorch::runtime::Result<size_t> size() const override {
    return size_;
  }

  ET_NODISCARD executorch::runtime::Error load_into(
      size_t offset,
      size_t size,
      ET_UNUSED const SegmentInfo& segment_info,
      void* buffer) const override {
    ET_CHECK_OR_RETURN_ERROR(
        buffer != nullptr,
        InvalidArgument,
        "Destination buffer cannot be null");

    auto result = load(offset, size, segment_info);
    if (!result.ok()) {
      return result.error();
    }
    std::memcpy(buffer, result->data(), size);
    return executorch::runtime::Error::Ok;
  }

 private:
  const uint8_t* const data_; // uint8 is easier to index into.
  const size_t size_;
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::BufferDataLoader;
} // namespace util
} // namespace executor
} // namespace torch
