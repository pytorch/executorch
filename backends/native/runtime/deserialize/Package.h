// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ByteSpan.h>
#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>
#include <executorch/backends/native/runtime/deserialize/ZipReader.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// Fixed member names inside a .ptn. The package survives being renamed because
// nothing depends on the file name.
constexpr const char* kProgramEntry = "program.ptg";
constexpr const char* kSafeTensorsEntry = "program.safetensors";
constexpr const char* kAliasesEntry = "aliases.json";

// One constant resolved out of a package.
struct Constant {
  ScalarType dtype = kFloat;
  const std::vector<int64_t>* sizes = nullptr;
  ByteSpan bytes;
  // Key that actually owns these bytes. Differs from the requested key when the
  // package deduplicated two byte-identical immutable constants.
  std::string owner;
};

// A loaded .ptn package: the serialized native Program plus the constants it
// references.
//
// Owns the package bytes; the zip and safetensors readers, and every span
// handed out, alias into that one buffer. std::vector's move preserves the
// buffer address, so those views stay valid across a move of the Package. Copy
// is deleted: a package is model-sized.
class Package {
 private:
  std::vector<uint8_t> bytes_;
  ZipReader zip_;
  ByteSpan program_;
  // Absent when the program references no constants, in which case the package
  // has no safetensors member at all.
  std::optional<SafeTensorsReader> tensors_;
  std::unordered_map<std::string, std::string> aliases_;

  Package() = default;

 public:
  ~Package() = default;
  Package(Package&&) noexcept = default;
  Package& operator=(Package&&) noexcept = default;
  Package(const Package&) = delete;
  Package& operator=(const Package&) = delete;

  // Parse a .ptn image. Takes ownership rather than copying, so a
  // hundred-megabyte package is resident once. Throws std::runtime_error if the
  // zip, the safetensors index, or the alias map is malformed, or if the
  // required program member is missing.
  static Package load(std::vector<uint8_t> bytes);

  // Read and parse a .ptn from disk. Throws std::runtime_error if the file
  // cannot be read.
  static Package load_file(const std::string& path);

  // The serialized native Program flatbuffer (the program.ptg member).
  ByteSpan program_bytes() const {
    return program_;
  }

  // Zip member names present, in central-directory order. Diagnostic only.
  const std::vector<std::string>& member_names() const {
    return zip_.names();
  }

  // Keys that own their bytes, in safetensors header order.
  const std::vector<std::string>& owner_keys() const;

  // Duplicate key -> owner key.
  const std::unordered_map<std::string, std::string>& aliases() const {
    return aliases_;
  }

  // Constant for `key`, resolving an alias to its owner. nullopt when the
  // package holds no such constant.
  std::optional<Constant> constant(const std::string& key) const;

  // Every key the package resolves, owners and aliases alike, sorted.
  std::vector<std::string> keys() const;

  // Total bytes across owner entries, i.e. what the constants actually cost.
  size_t constant_bytes() const;

  // True if `bytes` starts with the zip local-header signature, i.e. looks like
  // a package rather than a bare .ptg flatbuffer. Lets a tool accept either.
  static bool looks_like_package(ByteSpan bytes);
};

} // namespace ptn
