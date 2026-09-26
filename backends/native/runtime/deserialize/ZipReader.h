// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>

namespace ptn {

// Read-only access to stored members of a zip archive.
class ZipReader {
 private:
  struct Entry {
    uint64_t index = 0;
    size_t size = 0;
  };

  struct StringHash {
    using is_transparent = void;

    size_t operator()(std::string_view value) const noexcept {
      return std::hash<std::string_view>{}(value);
    }
  };

  struct Impl;

  std::unique_ptr<Impl> impl_;
  std::unordered_map<std::string, Entry, StringHash, std::equal_to<>> entries_;
  std::vector<std::string> names_;

  explicit ZipReader(std::unique_ptr<Impl> impl);
  const Entry* find_entry(std::string_view name) const;
  void read_entry_into(
      std::string_view name,
      const Entry& entry,
      size_t offset,
      MutableByteSpan destination) const;

 public:
  ~ZipReader();
  ZipReader(ZipReader&&) noexcept;
  ZipReader& operator=(ZipReader&&) noexcept;
  ZipReader(const ZipReader&) = delete;
  ZipReader& operator=(const ZipReader&) = delete;

  // Opens an archive without loading its member payloads.
  static ZipReader open(const std::string& path);

  // Opens an archive over caller-owned memory. `archive` must outlive this
  // reader and every read made through it.
  static ZipReader open(ByteSpan archive);

  // Member size, or nullopt when the member is absent.
  std::optional<size_t> member_size(std::string_view name) const;

  // Copies one complete member.
  std::vector<uint8_t> read(std::string_view name) const;

  // Copies a member range directly into caller-owned storage.
  void read_into(
      std::string_view name,
      size_t offset,
      MutableByteSpan destination) const;

  // Reads a complete member and verifies its checksum without retaining it.
  void verify(std::string_view name) const;

  const std::vector<std::string>& names() const {
    return names_;
  }
};

} // namespace ptn
