// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn::testing {

// Numerics a driver lets an engine trade away. Drivers then judge the engine as
// having made the trade, so an engine that never makes it rejects the option.
struct HostOptions {
  // Quantize the activations of linears whose weights alone are quantized.
  bool allow_activation_quant_for_weight_only = false;
};

// Defined once per binary by an engine's host library; drivers such as
// ptn_runner_main.cpp run every engine through the host it returns.
std::unique_ptr<EngineHost> make_engine_host(const HostOptions& options);

} // namespace ptn::testing
