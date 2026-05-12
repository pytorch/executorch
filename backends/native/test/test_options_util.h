/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Tiny shared helper for backends/native test binaries: build a
 * LoadBackendOptionsMap that forwards the NATIVE_COMPUTE_UNIT env var
 * to NativeBackend's "compute_unit" load-time option. Returns an empty
 * map when the env var is unset/empty/"auto" (the backend default).
 *
 * Usage:
 *   auto opts = load_options_for_compute_unit();
 *   module.load(opts);
 */

#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace native_test_util {

inline ::executorch::runtime::LoadBackendOptionsMap
load_options_for_compute_unit() {
  ::executorch::runtime::LoadBackendOptionsMap opts;
  const char* unit = std::getenv("NATIVE_COMPUTE_UNIT");
  if (!unit || !*unit || std::strcmp(unit, "auto") == 0) {
    return opts;
  }
  printf("  compute_unit (env): %s\n", unit);
  // Storage must outlive the view; keep it function-local-static so the
  // BackendOptions char array doesn't dangle when this returns.
  static ::executorch::runtime::BackendOptions<2> backend_opts;
  backend_opts = ::executorch::runtime::BackendOptions<2>{};
  if (backend_opts.set_option("compute_unit", unit) !=
      ::executorch::runtime::Error::Ok) {
    fprintf(stderr, "WARN: set_option(compute_unit) failed\n");
    return opts;
  }
  if (opts.set_options("NativeBackend", backend_opts.view()) !=
      ::executorch::runtime::Error::Ok) {
    fprintf(stderr, "WARN: set_options for NativeBackend failed\n");
  }
  return opts;
}

} // namespace native_test_util
