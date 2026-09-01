// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declaration of the generated FlatBuffer root type, included only from
// .cpp files so flatbuffers stays an implementation detail of the reader.
namespace native_backend {
struct Program;
} // namespace native_backend

namespace ptn {

namespace fbs = ::native_backend;

// Represents a loaded native-graph program
class Program {
 private:
  // Owns the bytes; the program_fb_ pointer aliases into this buffer.
  // std::vector's move preserves the buffer address, so program_fb_ stays valid
  // across a move. Never null: load() is the only constructor path and throws
  // rather than return a null root, so accessors dereference it unchecked.
  std::vector<uint8_t> bytes_;
  const fbs::Program* program_fb_ = nullptr;

  Program(std::vector<uint8_t> bytes, const fbs::Program* program_fb)
      : bytes_(std::move(bytes)), program_fb_(program_fb) {}

 public:
  ~Program() = default;
  Program(Program&&) noexcept = default;
  Program& operator=(Program&&) noexcept = default;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  // Parse and verify serialized native-graph bytes (a *.ptg buffer). Throws
  // std::runtime_error on failure.
  static Program load(const void* data, size_t size);

  const fbs::Program* flatbuffer() const {
    return program_fb_;
  }

  size_t num_methods() const;
};

} // namespace ptn
