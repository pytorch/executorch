// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The ggml engine host for PTN test drivers.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngineFlags.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>

namespace ptn::testing {

// cppcheck-suppress unusedFunction
std::unique_ptr<EngineHost> make_engine_host(const HostOptions& host_options) {
  GgmlEngineOptions options = ggml_engine_options_from_flags();
  options.allow_activation_quant_for_weight_only =
      host_options.allow_activation_quant_for_weight_only;
  return GgmlEngineHost::create(options);
}

} // namespace ptn::testing
