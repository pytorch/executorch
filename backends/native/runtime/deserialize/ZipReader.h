// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>

namespace ptn {

// Read-only reader for a stored (uncompressed) zip archive, which is what a
// .ptn package is. Deliberately not a general zip implementation: it serves
// whole members by name and rejects anything it cannot serve exactly, rather
// than degrading.
//
// Member payloads are located through the central directory, never by scanning
// local headers. The central directory is the authoritative record of sizes and
// offsets; a member written with an unknown size up front carries a data
// descriptor and a zeroed local header, which a local-header scan would
// misread.
class ZipReader {
 private:
  std::unordered_map<std::string, ByteSpan> members_;
  std::vector<std::string> names_;

 public:
  // Parse `archive`'s central directory. Throws std::runtime_error if it is not
  // a zip, is truncated, or holds a member this reader cannot serve
  // (compressed, encrypted, or with a payload outside the buffer).
  //
  // The returned archive borrows `archive`; it must outlive the ZipReader and
  // every span obtained from it.
  static ZipReader open(ByteSpan archive);

  // Payload of the named member, or nullopt if the archive has no such member.
  std::optional<ByteSpan> member(const std::string& name) const;

  // Member names, in central-directory order.
  const std::vector<std::string>& names() const {
    return names_;
  }
};

} // namespace ptn
