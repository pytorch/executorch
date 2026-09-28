/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_data/buffer_data_writer.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

using executorch::runtime::Error;
using executorch::runtime::Span;

namespace executorch {
namespace extension {
namespace {

bool add_overflows(size_t lhs, size_t rhs, size_t* result) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return true;
  }
  *result = lhs + rhs;
  return false;
}

} // namespace

BufferDataWriter::BufferDataWriter(std::vector<uint8_t>* output)
    : output_(output) {}

Error BufferDataWriter::write(Span<const uint8_t> data, size_t offset) {
  ET_CHECK_OR_RETURN_ERROR(
      output_ != nullptr, InvalidArgument, "Output vector cannot be null");
  ET_CHECK_OR_RETURN_ERROR(
      !published_, InvalidState, "Cannot write after publication");
  size_t end;
  ET_CHECK_OR_RETURN_ERROR(
      !add_overflows(offset, data.size(), &end),
      InvalidArgument,
      "Output range overflows size_t");
  if (pending_output_.size() < end) {
    pending_output_.resize(end, 0);
  }
  if (!data.empty()) {
    std::memcpy(pending_output_.data() + offset, data.data(), data.size());
  }
  return Error::Ok;
}

Error BufferDataWriter::publish() {
  ET_CHECK_OR_RETURN_ERROR(
      output_ != nullptr, InvalidArgument, "Output vector cannot be null");
  ET_CHECK_OR_RETURN_ERROR(
      !published_, InvalidState, "Writer has already published");
  *output_ = std::move(pending_output_);
  published_ = true;
  return Error::Ok;
}

} // namespace extension
} // namespace executorch
