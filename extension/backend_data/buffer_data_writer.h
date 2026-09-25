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
#include <vector>

#include <executorch/extension/backend_data/data_writer.h>

namespace executorch {
namespace extension {

/**
 * A DataWriter that publishes into a caller-owned byte vector.
 *
 * Writes are accumulated in private memory. The caller must keep `output`
 * alive for the writer's lifetime, but its old contents remain unchanged until
 * publish() succeeds. Destruction before publication discards pending bytes.
 */
class BufferDataWriter final : public DataWriter {
 public:
  /**
   * Creates a writer that will publish into `output`.
   *
   * @param[in,out] output Non-null caller-owned destination vector.
   */
  explicit BufferDataWriter(std::vector<uint8_t>* output);

  /** Synchronously copies `data` into a private buffer at `offset`. */
  ET_NODISCARD executorch::runtime::Error write(
      executorch::runtime::Span<const uint8_t> data,
      size_t offset) override;

  /** Moves the complete private output into the caller-owned vector. */
  ET_NODISCARD executorch::runtime::Error publish() override;

 private:
  std::vector<uint8_t>* output_;
  std::vector<uint8_t> pending_output_;
  bool published_{false};
};

} // namespace extension
} // namespace executorch
