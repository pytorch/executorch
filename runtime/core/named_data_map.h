/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
/**
 * Interface to access and retrieve data via name.
 * See executorch/extension/flat_tensor/ for an example.
 */
class NamedDataMap {
 public:
  virtual ~NamedDataMap() = default;
  /**
   * Get tensor_layout by key.
   *
   * @param key The name of the tensor.
   * @return Result containing TensorLayout.
   */
  ET_NODISCARD virtual Result<const TensorLayout> get_tensor_layout(
      executorch::aten::string_view key) const = 0;
  /**
   * Get data by key.
   *
   * @param key Name of the data.
   * @return Result containing a FreeableBuffer.
   */
  ET_NODISCARD virtual Result<FreeableBuffer> get_data(
      executorch::aten::string_view key) const = 0;

  /**
   * Loads data corresponding to the key into the provided buffer.
   *
   * @param key The name of the data.
   * @param size The number of bytes to load. Use `get_tensor_layout` to
   * retrieve the size of the data for a given key.
   * @param buffer The buffer to load the data into. Must point to at least
   * `size` bytes of memory.
   * @returns an Error indicating if the load was successful.
   */
  ET_NODISCARD virtual Error load_data_into(
      executorch::aten::string_view key,
      void* buffer,
      size_t size) const = 0;

  /**
   * Get the number of keys in the NamedDataMap.
   *
   * @return Result containing the number of keys.
   */
  ET_NODISCARD virtual Result<uint32_t> get_num_keys() const = 0;

  /**
   * Get the key at the given index.
   *
   * @param index The index of the key to retrieve.
   * @return Result containing the key at the given index. Note: the returned
   * pointer is only valid for the lifetime of the DataMap.
   */
  ET_NODISCARD virtual Result<const char*> get_key(uint32_t index) const = 0;
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
