/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/ir/GraphTypes.h>
#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/ir/Plan.h>
#include <executorch/backends/native/core/Runtime.h>

#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace backends {
namespace native {

struct RouterOptions {
  bool dump_trace = false;
};

/**
 * Router::route maps (Graph, Providers, Instances) -> Plan.
 *
 * Default implementation in routers/GreedyRouter.h. See §4.10 of the
 * design doc.
 */
class Router {
 public:
  virtual ~Router() = default;

  virtual ::executorch::runtime::Result<Plan> route(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<Runtime* const> providers,
      ::executorch::runtime::Span<Engine* const> instances,
      const ::executorch::runtime::NamedDataMap* ndm, // for constant validation; uploads are driven post-route by NativeBackend
      const RouterOptions& options) = 0;
};

} // namespace native
} // namespace backends
} // namespace executorch
