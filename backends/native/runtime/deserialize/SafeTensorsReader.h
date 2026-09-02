// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// One tensor's entry in a safetensors index.
struct TensorEntry {
  ScalarType dtype = kFloat;
  std::vector<int64_t> sizes;
  // Byte range within the blob's data section, not the whole blob.
  size_t offset = 0;
  size_t nbytes = 0;
};

// Reader for the safetensors format:
//
//     [u64 header_len][JSON header][data section]
//
// The header maps a tensor name to its dtype, shape, and byte range within the
// data section. The reserved "__metadata__" member is skipped.
//
// Tensor payloads are packed with no per-tensor padding, so an entry's absolute
// alignment within the file is arbitrary: copy through these spans rather than
// handing them to an API that requires alignment.
class SafeTensorsReader {
 private:
  // The data section only, i.e. the blob past its header.
  ByteSpan data_;
  std::unordered_map<std::string, TensorEntry> entries_;
  std::vector<std::string> names_;

 public:
  // Parse `blob`'s index. Throws std::runtime_error if the blob is truncated,
  // the header is not a JSON object, a dtype has no ScalarType, or a byte range
  // is inconsistent with its dtype and shape.
  //
  // Borrows `blob`, which must outlive both this reader and any span from it.
  static SafeTensorsReader open(ByteSpan blob);

  // Entry for `name`, or nullptr when absent.
  const TensorEntry* find(const std::string& name) const;

  // Payload of an entry obtained from this reader.
  ByteSpan bytes(const TensorEntry& entry) const;

  // Tensor names, in header order, excluding "__metadata__".
  const std::vector<std::string>& names() const {
    return names_;
  }

  // Total payload bytes across all entries.
  size_t total_bytes() const;
};

} // namespace ptn
