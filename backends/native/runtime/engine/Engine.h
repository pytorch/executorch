// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

struct Method;
class Package;
class Program;

// A backend-compiled region of a Method. Full delegation produces one per
// Method; partitioning may produce several. Not thread-safe or re-entrant.
// Executables from the same context may alias mutable state and must be
// serialized; executables from different contexts may run concurrently.
class EngineExecutable {
 protected:
  EngineExecutable() = default;

 public:
  EngineExecutable(const EngineExecutable&) = delete;
  EngineExecutable& operator=(const EngineExecutable&) = delete;
  EngineExecutable(EngineExecutable&&) = delete;
  EngineExecutable& operator=(EngineExecutable&&) = delete;
  virtual ~EngineExecutable();

  // Counts, dtypes, and static shapes of user inputs and outputs in graph
  // order.
  virtual size_t num_inputs() const = 0;
  virtual size_t num_outputs() const = 0;
  virtual std::vector<int64_t> input_sizes(size_t i) const = 0;
  virtual std::vector<int64_t> output_sizes(size_t i) const = 0;
  virtual ScalarType input_dtype(size_t i) const = 0;
  virtual ScalarType output_dtype(size_t i) const = 0;

  // Copy `numel` elements from host `data` into input i, converting from
  // `src_dtype` to the input's dtype when they differ. `numel` must equal the
  // input's element count; `data` may be null only when `numel` is zero.
  virtual void
  set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype) = 0;

  // Run the compiled Method. Blocks until every output is readable, so a
  // get_output right after it needs no further synchronization.
  virtual void execute() = 0;

  // Copy output i back into host `data`, converting to `dst_dtype` from the
  // output's dtype when they differ. `data` may be null only when `numel` is
  // zero. Only meaningful after an execute().
  virtual void
  get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype) = 0;
};

// Engine-owned state for one loaded Program. Matching non-empty DataBinding
// keys share storage within a context; contexts never share mutable buffers.
//
// Owns its Program and Package and must outlive its executables. Compilation
// and execution through one context are not thread-safe.
class EngineContext {
 protected:
  EngineContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package);

  const Program& program() const;
  const Package& package() const;

  // Implement the backend-specific lowering for `method`. Return a non-null
  // executable or throw; returning null violates the interface contract.
  virtual std::unique_ptr<EngineExecutable> compile_method(
      const Method& method) = 0;

 public:
  EngineContext(const EngineContext&) = delete;
  EngineContext& operator=(const EngineContext&) = delete;
  EngineContext(EngineContext&&) = delete;
  EngineContext& operator=(EngineContext&&) = delete;
  virtual ~EngineContext();

  // Compile the named Method and prepack its constants. Returns non-null or
  // throws std::runtime_error for an absent, invalid, or unsupported method.
  std::unique_ptr<EngineExecutable> compile(const std::string& method_name);

 private:
  std::shared_ptr<const Program> program_;
  std::shared_ptr<const Package> package_;
};

// Process-wide backend and device resources shared by its contexts. Must
// outlive every context and executable it creates.
// Callers serialize create_context(); distinct contexts may execute
// concurrently, so process-wide resources must support that concurrency.
class EngineHost {
 protected:
  EngineHost() = default;

 public:
  EngineHost(const EngineHost&) = delete;
  EngineHost& operator=(const EngineHost&) = delete;
  EngineHost(EngineHost&&) = delete;
  EngineHost& operator=(EngineHost&&) = delete;
  virtual ~EngineHost();

  // Diagnostic backend and device names; neither controls dispatch.
  virtual const std::string& name() const = 0;
  virtual const std::string& device_name() const = 0;

  // Create an isolated state domain, sharing ownership of `program` and
  // `package`. Returns non-null or throws on invalid or unsupported resources.
  virtual std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) = 0;
};

} // namespace ptn
