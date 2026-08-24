// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Forward-declaration of the generated FlatBuffer root type. The generated
// header is included only in Program.cpp / utils/ToGraphViz.cpp, so flatbuffers
// stays an implementation detail of the reader.
namespace native_backend {
struct Program;
} // namespace native_backend

namespace ptn {

// A loaded native-graph program: owns the serialized FlatBuffer bytes and
// exposes a zero-copy view of the root. Reader-only for now (no in-memory
// mutable graph, constants, or execution).
class Program {
 private:
  // Owns the bytes; the program_fb_ pointer aliases into this buffer.
  // std::vector's move preserves the buffer address, so program_fb_ stays valid
  // across a move. Never null on a live Program: the constructor is private and
  // load(), the only caller, throws rather than hand back a null root, so the
  // accessors below dereference it unchecked.
  std::vector<uint8_t> bytes_;
  const ::native_backend::Program* program_fb_ = nullptr;

  Program(
      std::vector<uint8_t> bytes,
      const ::native_backend::Program* program_fb)
      : bytes_(std::move(bytes)), program_fb_(program_fb) {}

 public:
  ~Program() = default;
  Program(Program&&) noexcept = default;
  Program& operator=(Program&&) noexcept = default;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  // Parse and verify serialized native-graph bytes (a *.nptg buffer). Throws
  // std::runtime_error on failure. The returned Program owns a copy of the
  // bytes; the zero-copy accessors are valid for its lifetime.
  static Program load(const void* data, size_t size);

  // Zero-copy FlatBuffer root, pointing into this Program's owned bytes.
  const ::native_backend::Program* flatbuffer() const {
    return program_fb_;
  }

  size_t num_methods() const;

  // Render this program to Graphviz DOT text (impl in utils/ToDot.cpp). Pure
  // string builder; the caller writes/renders it (e.g. `dot -Tpng`).
  std::string to_dot() const;
};

} // namespace ptn
