/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

// The sampler utility requires the dtype-dispatch macros defined above.
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor_ptr.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Samples a flattened row through a non-owning [vocab] view. Requires dense,
// row-major CPU logits and a sampler with matching vocabulary size.
ET_EXPERIMENTAL inline std::optional<Token>
sample_from_row(const aten::Tensor& logits, int64_t row, Sampler& sampler) {
  if (logits.dim() <= 0 || row < 0) {
    return std::nullopt;
  }
  const auto vocab_size = logits.size(logits.dim() - 1);
  if (vocab_size <= 0 ||
      vocab_size > std::numeric_limits<aten::SizesType>::max() ||
      row >= logits.numel() / vocab_size) {
    return std::nullopt;
  }
  auto one_row = make_tensor_ptr(
      {static_cast<aten::SizesType>(vocab_size)},
      static_cast<uint8_t*>(logits.mutable_data_ptr()) +
          static_cast<std::size_t>(row) * vocab_size *
              runtime::elementSize(logits.scalar_type()),
      logits.scalar_type());
  const auto token = sample_from_logits(*one_row, sampler);
  if (token < 0 || token >= vocab_size) {
    return std::nullopt;
  }
  return static_cast<Token>(token);
}

// Owns one newly allocated sequence until publication. The control must outlive
// the guard, and seq_rm must not throw, including during exception unwinding.
class ET_EXPERIMENTAL SequenceGuard final {
 public:
  SequenceGuard(cache::BatchControl& control, int32_t seq)
      : control_(control), seq_(seq) {}

  SequenceGuard(const SequenceGuard&) = delete;
  SequenceGuard& operator=(const SequenceGuard&) = delete;
  SequenceGuard(SequenceGuard&&) = delete;
  SequenceGuard& operator=(SequenceGuard&&) = delete;

  ~SequenceGuard() {
    if (owned_) {
      control_.seq_rm(seq_);
    }
  }

  void release() noexcept {
    owned_ = false;
  }

 private:
  cache::BatchControl& control_;
  int32_t seq_;
  bool owned_ = true;
};

// insert constructs the executor's state and publishes it atomically: false or
// an exception must leave no new session behind. Only true transfers ownership.
template <class Insert>
ET_EXPERIMENTAL std::optional<SessionId> publish_sequence(
    cache::BatchControl& control,
    int32_t seq,
    Position expected,
    SessionId& next,
    Insert&& insert) {
  SequenceGuard guard(control, seq);
  if (next == 0 || control.pos(seq) != expected) {
    return std::nullopt;
  }
  const SessionId session = next;
  if (!std::forward<Insert>(insert)(session, seq)) {
    return std::nullopt;
  }
  guard.release();
  next = session == std::numeric_limits<SessionId>::max() ? 0 : session + 1;
  return session;
}

// The caller owns both the module and the installed cache. The option storage
// stays live throughout load_method(), where backend initialization resolves
// it.
template <class ModuleLike>
ET_EXPERIMENTAL runtime::Error load_method_with_cache(
    ModuleLike& module,
    const std::string& method,
    const char* backend,
    const cache::InstallGuard& install_guard) {
  runtime::BackendOptions<1> options;
  const auto option_error = install_guard.set_option(options);
  if (option_error != runtime::Error::Ok) {
    return option_error;
  }
  runtime::LoadBackendOptionsMap map;
  const auto map_error = map.set_options(backend, options.view());
  if (map_error != runtime::Error::Ok) {
    return map_error;
  }
  return module.load_method(method, nullptr, nullptr, &map);
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
