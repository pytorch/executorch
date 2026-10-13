// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The ggml engine for ptn_runner_main.cpp.

#include <stdexcept>

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>
#include <gflags/gflags.h>

DEFINE_string(
    device,
    "cpu",
    "ggml device: cpu, or gpu for the GPU backend this binary links.");
DEFINE_int32(device_index, 0, "GPU device index.");
DEFINE_int32(cpu_threads, 0, "ggml CPU threads; 0 keeps ggml's default.");
DEFINE_int32(
    max_cached_graphs,
    1,
    "Graphs kept per executable by input shapes; 0 rebuilds every call.");
DEFINE_bool(
    print_graph_ops,
    false,
    "Print each built graph's node count per ggml op to stderr.");

namespace ptn::testing {

std::unique_ptr<EngineHost> make_engine_host() {
  GgmlEngineOptions options;
  if (FLAGS_device == "gpu") {
    options.device = GgmlDevice::Gpu;
  } else if (FLAGS_device != "cpu") {
    throw std::invalid_argument("--device must be cpu or gpu");
  }
  options.device_index = FLAGS_device_index;
  options.cpu_threads = FLAGS_cpu_threads;
  if (FLAGS_max_cached_graphs < 0) {
    throw std::invalid_argument("--max_cached_graphs must be non-negative");
  }
  options.max_cached_graphs = static_cast<size_t>(FLAGS_max_cached_graphs);
  options.print_graph_ops = FLAGS_print_graph_ops;
  return GgmlEngineHost::create(options);
}

} // namespace ptn::testing
