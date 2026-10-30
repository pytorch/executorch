// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

struct ConstantBytes {
  size_t unique = 0;
  size_t materialized = 0;
};

template <typename Context>
ConstantBytes constant_bytes(const EngineContext& context) {
  const auto& typed = dynamic_cast<const Context&>(context);
  return {typed.unique_constant_bytes(), typed.materialized_constant_bytes()};
}

int run_classification(
    const std::function<std::unique_ptr<EngineHost>()>& make_host,
    const std::function<ConstantBytes(const EngineContext&)>& constant_bytes);

} // namespace ptn
