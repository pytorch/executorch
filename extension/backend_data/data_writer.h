/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {

/**
 * Builds and publishes one complete replacement for one physical data source.
 *
 * write() calls describe bytes in a hypothetical output and do not make them
 * externally visible. The caller may issue writes in any order. One writer
 * instance supports one write-to-publish operation. A concrete implementation
 * must discard unpublished output from its destructor.
 */
class DataWriter {
 public:
  virtual ~DataWriter() = default;

  /**
   * Synchronously copies bytes into the unpublished output.
   *
   * `data` only needs to remain valid until this call returns. The range begins
   * at the absolute byte `offset` from the start of the output. Calls may be
   * unordered. If ranges overlap, later writes replace earlier bytes.
   *
   * A zero-length write is valid and extends the logical output size to at
   * least `offset`; callers can therefore represent trailing zero-filled
   * space without allocating a buffer for it.
   *
   * @param[in] data Bytes to copy into writer-owned storage.
   * @param[in] offset Absolute byte offset in the hypothetical output.
   */
  ET_NODISCARD virtual executorch::runtime::Error write(
      executorch::runtime::Span<const uint8_t> data,
      size_t offset) = 0;

  /**
   * Makes all written ranges durable and atomically exposes the output.
   *
   * A successful call publishes exactly the range from byte zero through the
   * greatest end offset observed by write(). Calling publish() more than once,
   * or calling write() after successful publication, is an error.
   */
  ET_NODISCARD virtual executorch::runtime::Error publish() = 0;
};

} // namespace extension
} // namespace executorch
