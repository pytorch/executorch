// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/backends/native/runtime/MethodMeta.h>
#include <executorch/runtime/executor/method_meta.h>

namespace executorch::extension::native_module {

class MethodMetaBridge final {
 public:
  static std::unique_ptr<MethodMetaBridge> create(const ptn::MethodMeta& meta);

  ~MethodMetaBridge() = default;
  MethodMetaBridge(const MethodMetaBridge&) = delete;
  MethodMetaBridge& operator=(const MethodMetaBridge&) = delete;
  MethodMetaBridge(MethodMetaBridge&&) = delete;
  MethodMetaBridge& operator=(MethodMetaBridge&&) = delete;

  // The returned view borrows this bridge's storage.
  ET_RUNTIME_NAMESPACE::MethodMeta view() const {
    return view_;
  }

 private:
  explicit MethodMetaBridge(std::vector<uint8_t> bytes);

  // `bytes_` must precede `view_`, which borrows its storage.
  std::vector<uint8_t> bytes_;
  ET_RUNTIME_NAMESPACE::MethodMeta view_;
};

} // namespace executorch::extension::native_module
