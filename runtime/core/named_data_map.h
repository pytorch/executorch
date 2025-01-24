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
namespace runtime {

/**
 * Interface to access and retrieve data via name.
 * See executorch/extension/flat_tensor/ for an example.
 */
class ET_EXPERIMENTAL NamedDataMap {
 public:
  virtual ~NamedDataMap() = default;
  /**
   * Get tensor metadata by fully qualified name (FQN).
   *
   * @param fqn Fully qualified name of the tensor.
   * @return Result containing TensorLayout with tensor metadata.
   */
  ET_NODISCARD virtual Result<const executorch::runtime::TensorLayout>
  get_metadata(const char* fqn) const = 0;
  /**
   * Get tensor data by fully qualified name (FQN).
   *
   * @param fqn Fully qualified name of the tensor.
   * @return Result containing a FreeableBuffer with the tensor data.
   */
  ET_NODISCARD virtual Result<FreeableBuffer> get_data(
      const char* fqn) const = 0;

  /**
   * Loads data corresponding to the fqn into the provided buffer.
   *
   * @param fqn Fully qualified name of the tensor.
   * @param size The number of bytes to load. Use `get_metadata` to retrieve the
   * size of the tensor for a given fqn.
   * @param buffer The buffer to load the data into. Must point to at least
   * `size` bytes of memory.
   * @return Result containing the number of bytes written on success.
   */
  ET_NODISCARD virtual Result<size_t>
  load_data_into(const char* fqn, size_t size, void* buffer) const = 0;

  /**
   * Get the number of keys in the NamedDataMap.
   *
   * @return Result containing the number of keys.
   */
  ET_NODISCARD virtual Result<size_t> get_num_keys() const = 0;

  /**
   * Get the key at the given index.
   *
   * @param index The index of the key to retrieve.
   * @return Result containing the key at the given index. Note: the returned
   * pointer is only valid for the lifetime of the DataMap.
   */
  ET_NODISCARD virtual Result<const char*> get_key(size_t index) const = 0;
};

} // namespace runtime
} // namespace executorch
