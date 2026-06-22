/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace pybindings {

/// DataLoader wrapper holding a shared_ptr, allowing sharing between Python
/// and C++ while Module takes ownership via unique_ptr.
class SharedPtrDataLoader final : public runtime::DataLoader {
 public:
  explicit SharedPtrDataLoader(std::shared_ptr<runtime::DataLoader> loader)
      : loader_(std::move(loader)) {}

  ET_NODISCARD runtime::Result<runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const SegmentInfo& segment_info) const override {
    return loader_->load(offset, size, segment_info);
  }

  ET_NODISCARD runtime::Result<size_t> size() const override {
    return loader_->size();
  }

  ET_NODISCARD runtime::Error load_into(
      size_t offset,
      size_t size,
      const SegmentInfo& segment_info,
      void* buffer) const override {
    return loader_->load_into(offset, size, segment_info, buffer);
  }

 private:
  std::shared_ptr<runtime::DataLoader> loader_;
};

/// Pybind11 wrapper for DataLoader. Use shared_ptr holder type in pybind11.
struct PyDataLoader {
  explicit PyDataLoader(std::shared_ptr<runtime::DataLoader> loader)
      : loader_(std::move(loader)) {}

  PyDataLoader(const PyDataLoader&) = delete;
  PyDataLoader& operator=(const PyDataLoader&) = delete;
  PyDataLoader(PyDataLoader&&) = default;
  PyDataLoader& operator=(PyDataLoader&&) = default;

  std::shared_ptr<runtime::DataLoader> get() const {
    return loader_;
  }

  /// Creates a unique_ptr DataLoader that delegates to the shared loader.
  std::unique_ptr<runtime::DataLoader> make_delegating_loader() const {
    return std::make_unique<SharedPtrDataLoader>(loader_);
  }

 private:
  std::shared_ptr<runtime::DataLoader> loader_;
};

} // namespace pybindings
} // namespace extension
} // namespace executorch
