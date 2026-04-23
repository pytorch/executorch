/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Instance.h>
#include <executorch/backends/portable/runtime_v2/api/Plan.h>
#include <executorch/backends/portable/runtime_v2/api/Provider.h>
#include <executorch/backends/portable/runtime_v2/api/GraphTypes.h>

#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace backends {
namespace portable_v2 {

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
      ::executorch::runtime::Span<Provider* const> providers,
      ::executorch::runtime::Span<Instance* const> instances,
      const ::executorch::runtime::NamedDataMap* ndm,  // for upload_constant
      const RouterOptions& options) = 0;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
