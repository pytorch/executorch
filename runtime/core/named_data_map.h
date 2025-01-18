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
 * Interface to access and retrieve data via name from a loaded data file.
 * See executorch/extension/flat_tensor/ for an example.
 */
class NamedDataMap {
 public:
  virtual ~NamedDataMap() = default;
  /**
   * Get tensor metadata by fully qualified name (FQN).
   *
   * @param fqn Fully qualified name of the tensor.
   * @return Result containing a pointer to the metadata.
   */
  ET_NODISCARD virtual Result<const executorch::runtime::TensorLayout>
  get_metadata(const char* fqn) const = 0;
  /**
   * Get tensor data by fully qualified name (FQN).
   *
   * @param fqn Fully qualified name of the tensor.
   * @return Result containing a span of uint8_t representing the tensor data.
   */
  ET_NODISCARD virtual Result<Span<const uint8_t>> get_data(
      const char* fqn) const = 0;
};

} // namespace runtime
} // namespace executorch
