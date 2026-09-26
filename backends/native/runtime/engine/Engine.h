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

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// One dependency-closed region of a Method, lowered onto a backend and ready to
// run: whatever the backend needed to compile it, plus the staging it reads
// inputs from and writes outputs to.
//
// Under full delegation -- the only mode today -- that region is the whole
// Method, and the executable's inputs and outputs are the Method's. Under
// runtime partitioning one Method yields several executables interleaved with
// other backends, and the inputs and outputs are region boundaries instead.
//
// Obtained from EngineContext::compile, never constructed directly. Not
// thread-safe and not re-entrant: one executable runs one call at a time.
// Concurrent inference means several executables (or, once the working-set
// split lands, several working sets over one compiled program).
class EngineExecutable {
 protected:
  EngineExecutable() = default;

 public:
  EngineExecutable(const EngineExecutable&) = delete;
  EngineExecutable& operator=(const EngineExecutable&) = delete;
  EngineExecutable(EngineExecutable&&) = delete;
  EngineExecutable& operator=(EngineExecutable&&) = delete;
  virtual ~EngineExecutable();

  // Counts and shapes of the compiled Method's user inputs / outputs, in graph
  // order. Sizes are static upper bounds, so a dynamic dim reports its maximum.
  virtual size_t num_inputs() const = 0;
  virtual size_t num_outputs() const = 0;
  virtual std::vector<int64_t> input_sizes(size_t i) const = 0;
  virtual std::vector<int64_t> output_sizes(size_t i) const = 0;
  virtual ScalarType input_dtype(size_t i) const = 0;
  virtual ScalarType output_dtype(size_t i) const = 0;

  // Copy `numel` elements from host `data` into input i, converting from
  // `src_dtype` to the input's dtype when they differ. `numel` must equal the
  // input's element count.
  virtual void
  set_input(size_t i, const void* data, size_t numel, ScalarType src_dtype) = 0;

  // Run the compiled Method. Blocks until every output is readable, so a
  // get_output right after it needs no further synchronization.
  virtual void execute() = 0;

  // Copy output i back into host `data`, converting to `dst_dtype` from the
  // output's dtype when they differ. Only meaningful after an execute().
  virtual void
  get_output(size_t i, void* data, size_t numel, ScalarType dst_dtype) = 0;
};

// A compute backend, at process scope: the device context and kernel registry
// that every Method run on that device shares. One per device, constructed
// through the backend's own factory (e.g. make_vulkan_engine()) since selecting
// a backend is the caller's decision, not this interface's.
//
// Must outlive every executable it compiled.
class EngineContext {
 protected:
  EngineContext() = default;

 public:
  EngineContext(const EngineContext&) = delete;
  EngineContext& operator=(const EngineContext&) = delete;
  EngineContext(EngineContext&&) = delete;
  EngineContext& operator=(EngineContext&&) = delete;
  virtual ~EngineContext();

  // Backend identity ("vulkan"), and the device it selected ("SwiftShader
  // Device"). Diagnostics only; nothing dispatches on either.
  virtual const std::string& name() const = 0;
  virtual const std::string& device_name() const = 0;

  // Lower `method` onto this backend and prepack the constants it binds,
  // fetched from `package` by data_key. `method` must outlive the returned
  // executable; `package` is needed only for this call.
  //
  // Compiles the method whole, which is full delegation -- the only mode today.
  // Runtime partitioning narrows the unit to a region of a method and yields
  // several executables per method; that arrives as an added entry point, not a
  // change to this one.
  //
  // Throws std::runtime_error when the backend cannot run the method: an
  // unsupported op or dtype, a binding whose constant the package does not
  // hold, a constant whose byte count contradicts its TensorMeta, an unbounded
  // dynamic dim, or a higher-order-op subgraph. A backend is free to reject
  // anything else it cannot lower; there is no partial success.
  virtual std::unique_ptr<EngineExecutable> compile(
      const Method& method,
      const Package& package) = 0;
};

} // namespace ptn
