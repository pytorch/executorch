// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

enum class GgmlDevice {
  Cpu,
  // The GPU backend is whichever one the linked ggml variant registers (CUDA,
  // HIP, ...); device_index counts GPU devices only.
  Gpu,
};

struct GgmlEngineOptions {
  GgmlDevice device = GgmlDevice::Cpu;
  int device_index = 0;
  // Threads for the ggml CPU backend, which also runs ops the GPU backend
  // lacks. 0 keeps ggml's default of 4, not the host core count.
  int cpu_threads = 0;
  // Node capacity of each ggml graph and of the backend scheduler; ggml
  // aborts on a graph that outgrows it.
  size_t graph_size = 4096;
  // Graphs each executable keeps, keyed by the shapes of all inputs and, for
  // each attention mask that leaves only a key prefix open, that prefix
  // rounded up to 256 keys: a decode step crossing a 256-key boundary builds
  // a new graph. A call matching a cached key replays the graph with fresh
  // input bytes, skipping lowering and allocation; every entry holds its own
  // scheduler allocation. 0 rebuilds the graph on every call; at most 16.
  size_t max_cached_graphs = 1;
  // Prints each newly built graph's node count per ggml op to stderr, which
  // shows whether a lowering rewrite fired.
  bool print_graph_ops = false;
};

class GgmlEngineContext;

class GgmlEngineHost final : public EngineHost {
 public:
  static std::unique_ptr<GgmlEngineHost> create(GgmlEngineOptions options = {});

  ~GgmlEngineHost() override;

  const std::string& name() const override;
  const std::string& device_name() const override;

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override;

  std::unique_ptr<GgmlEngineContext> create_ggml_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package);

 private:
  struct Impl;

  explicit GgmlEngineHost(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class GgmlEngineContext final : public EngineContext {
 public:
  ~GgmlEngineContext() override;

  size_t materialized_constant_bytes() const;
  // Constant bytes copied onto the heap on their way into ggml buffers, rather
  // than read in place from the package.
  size_t heap_copied_constant_bytes() const;

 private:
  friend class GgmlEngineHost;

  struct Impl;

  GgmlEngineContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      GgmlEngineOptions options);

  std::unique_ptr<EngineExecutable> compile_method(
      const Method& method) override;

  std::unique_ptr<Impl> impl_;
};

} // namespace ptn
