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
#include <memory>

namespace executorch {
namespace extension {

/**
 * A DataLoader that wraps a pre-allocated buffer and shares ownership to it.
 * The FreeableBuffers that it returns do not actually free any data.
 *
 * This can be used to wrap data that was allocated elsewhere.
 */
class SharedPtrDataLoader : public executorch::runtime::DataLoader {
 public:
  SharedPtrDataLoader(std::shared_ptr<void> data, size_t size)
      : data_(data), size_(size) {}

  __ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      __ET_UNUSED const DataLoader::SegmentInfo& segment_info) override {
    ET_CHECK_OR_RETURN_ERROR(
        offset + size <= size_,
        InvalidArgument,
        "offset %zu + size %zu > size_ %zu",
        offset,
        size,
        size_);
    return executorch::runtime::FreeableBuffer(
        static_cast<uint8_t*>(data_.get()) + offset, size, /*free_fn=*/nullptr);
  }

  __ET_NODISCARD executorch::runtime::Result<size_t> size() const override {
    return size_;
  }

 private:
  const std::shared_ptr<void> data_;
  const size_t size_;
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::SharedPtrDataLoader;
} // namespace util
} // namespace executor
} // namespace torch
