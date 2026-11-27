// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <span>
#include <string_view>

#include <executorch/backends/native/runtime/api/Tensor.h>

namespace ptn {

namespace detail {
class SessionImpl;
} // namespace detail

// Mutable execution state for one Model. Methods share persistent state within
// a Session, while separate Sessions remain isolated. Not thread-safe or
// re-entrant; distinct Sessions may run concurrently when supported.
class Session final {
 public:
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  ~Session();

  // Compiles and caches `method_name`; a failure throws and leaves no entry.
  void prepare(std::string_view method_name);

  // Releases compiled method resources without resetting context-owned state.
  void release(std::string_view method_name);

  bool is_prepared(std::string_view method_name) const;

  // Throws on validation or engine failure. Outputs are copied only after
  // every engine read succeeds.
  void run(
      std::string_view method_name,
      std::span<const ConstTensorView> inputs,
      std::span<MutableTensorView> outputs);

 private:
  friend class Model;

  explicit Session(std::unique_ptr<detail::SessionImpl> impl);

  std::unique_ptr<detail::SessionImpl> impl_;
};

} // namespace ptn
