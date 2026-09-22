// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>
#include <executorch/backends/native/runtime/deserialize/ZipReader.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// Fixed member names inside a .ptn. The package survives being renamed because
// nothing depends on the file name.
constexpr const char* kProgramEntry = "program.ptg";
constexpr const char* kSafeTensorsEntry = "program.safetensors";
constexpr const char* kAliasesEntry = "aliases.json";

// Metadata for one constant resolved out of a package. `sizes` is borrowed from
// the Package and must not outlive it.
struct ConstantInfo {
  uint64_t package_id = 0;
  ScalarType dtype = kFloat;
  const std::vector<int64_t>* sizes = nullptr;
  size_t nbytes = 0;
  // Key that actually owns these bytes. Differs from the requested key when the
  // package deduplicated two byte-identical immutable constants.
  std::string owner;
};

// A loaded .ptn package: the serialized native Program plus the constants it
// references.
//
// Opening a file-backed package reads its directory and metadata, but leaves
// weight payloads on disk until an engine requests them.
class Package {
 private:
  uint64_t id_ = 0;
  OwnedBytes archive_bytes_;
  std::optional<ZipReader> zip_;
  std::vector<uint8_t> program_;
  // Absent when the program references no constants, in which case the package
  // has no safetensors member at all.
  std::optional<SafeTensorsReader> tensors_;
  size_t tensor_data_offset_ = 0;
  std::unordered_map<std::string, std::string> aliases_;

  Package();

 public:
  ~Package() = default;
  Package(Package&&) noexcept = default;
  Package& operator=(Package&& other) noexcept;
  Package(const Package&) = delete;
  Package& operator=(const Package&) = delete;

  // Open and parse the .ptn at `path` without loading its weight payloads.
  static Package load(const std::string& path);

  // Parse a .ptn image already in hand. Takes ownership rather than copying, so
  // a hundred-megabyte package is resident once. For callers that must inspect
  // the bytes before deciding this is a package at all; everyone else should
  // use the path overload. Throws std::runtime_error if the zip, the
  // safetensors index, or the alias map is malformed, or if the required
  // program member is missing.
  static Package load(OwnedBytes bytes);

  // The serialized native Program flatbuffer (the program.ptg member).
  ByteSpan program_bytes() const {
    return ByteSpan(program_);
  }

  // Zip member names present, in central-directory order. Diagnostic only.
  const std::vector<std::string>& member_names() const {
    return zip_->names();
  }

  // Keys that own their bytes, in safetensors header order.
  const std::vector<std::string>& owner_keys() const;

  // Duplicate key -> owner key.
  const std::unordered_map<std::string, std::string>& aliases() const {
    return aliases_;
  }

  // Metadata for `key`, resolving an alias to its owner. nullopt when absent.
  std::optional<ConstantInfo> constant_info(const std::string& key) const;

  // Load one constant into a new owning buffer. nullopt when absent.
  std::optional<OwnedBytes> acquire_constant(const std::string& key) const;

  // Load one constant directly into an exact-sized destination. Returns false
  // when absent and throws when the destination has the wrong size.
  bool load_constant_into(const std::string& key, MutableByteSpan destination)
      const;

  // Stream all weight bytes once to verify the zip member checksum.
  void verify_constants() const;

  // Every key the package resolves, owners and aliases alike, sorted.
  std::vector<std::string> keys() const;

  // Total bytes across owner entries, i.e. what the constants actually cost.
  size_t constant_bytes() const;

  // True if `bytes` starts with the zip local-header signature, i.e. looks like
  // a package rather than a bare .ptg flatbuffer. Lets a tool accept either.
  static bool looks_like_package(ByteSpan bytes);

 private:
  void load_metadata();
};

} // namespace ptn
