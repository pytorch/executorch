// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <executorch/backends/native/runtime/api/MethodInfo.h>

namespace ptn {

class EngineHost;

class Session;

namespace detail {
class ModelImpl;
} // namespace detail

enum class FileMode {
  Read,
  Mmap,
};

// An immutable, structurally validated PTN image. Copies share package bytes
// and metadata; metadata reads are thread-safe.
class Model final {
 public:
  Model(const Model&);
  Model& operator=(const Model&);
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  ~Model();

  // Throws if the source cannot be read or is not a valid PTN package.
  static Model load_file(std::string_view path, FileMode mode = FileMode::Mmap);

  // Consumes the buffer and throws if it is not a valid PTN package.
  static Model load_bytes(std::vector<uint8_t> bytes);

  // The returned view remains valid while this Model remains alive.
  std::span<const std::string> method_names() const;

  // Throws std::invalid_argument if the method is absent.
  MethodInfo method_info(std::string_view name) const;

  // Creates isolated mutable state. `host` is borrowed and must outlive the
  // Session. Concurrent creation requires support from the host; host
  // exceptions propagate to the caller.
  Session create_session(EngineHost& host) const;

 private:
  explicit Model(std::shared_ptr<const detail::ModelImpl> impl);

  std::shared_ptr<const detail::ModelImpl> impl_;
};

} // namespace ptn
