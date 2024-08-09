/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <executorch/runtime/backend/backend_execution_context.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {

struct SizedBuffer {
  void* buffer;
  size_t nbytes; // number of bytes of buffer
};

struct CompileSpec {
  const char* key; // spec key
  SizedBuffer value; // spec value
};

/**
 * An opaque handle managed by a backend. Typically points to a backend-private
 * class/struct.
 */
using DelegateHandle = void;

class PyTorchBackendInterface {
 public:
  virtual ~PyTorchBackendInterface() = 0;

  /**
   * Returns true if the backend is available to process delegation calls.
   */
  __ET_NODISCARD virtual bool is_available() const = 0;

  /**
   * Responsible to further process (compile/transform/optimize) the compiled
   * unit that was produced, ahead-of-time, as well as perform any backend
   * initialization to ready it for execution. This method is called every time
   * the PyTorch program is initialized. Consequently, this is the place to
   * perform any backend initialization as well as transformations,
   * optimizations, and even compilation that depend on the target device. As
   * such, it is strongly encouraged to push as much processing as possible to
   * the ahead-of-time processing.
   *
   * @param[in] processed An opaque (to PyTorch) compiled unit from the
   *     preprocessor. Can contain anything the backend needs to execute the
   *     equivalent semantics of the passed-in Module and its method. Often
   *     passed unmodified to `execute()` as a `DelegateHandle`, unless it needs
   *     further processing at init time to be fully executable. If the data is
   *     not needed after init(), calling processed->Free() can reclaim its
   *     memory.
   * @param[in] compile_specs The exact same compiler specification that
   *     was used ahead-of-time to produce `processed`.
   *
   * @returns On success, an opaque handle representing the the method
   *     implemented by the delegate. This handle is passed to `execute()` and
   *     `destroy()`, and the memory it points to is owned by the backend.
   *     Typically points to a backend-private class/struct.
   * @returns On error, returns an error code other than Error::Ok. If the
   *     compiled unit (the preprocessed result from ahead of time) is not
   *     compatible with the current backend runtime, return the error code
   *     Error::DelegateInvalidCompatibility. Other backend delegate
   *     specific error codes can be found in error.h.
   */
  __ET_NODISCARD virtual Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const = 0;

  /**
   * Responsible for executing the given method’s handle, as it was produced
   * by compile.
   *
   * @param[in] handle An opaque handle returned by `init()`. Usually a backend
   *     executable unit. This executable unit should be ready to execute the
   *     delegate blobs.
   * @param[in] args The method’s inputs and outputs.
   * @retval Error::Ok if successful.
   */
  __ET_NODISCARD virtual Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const = 0;

  /**
   * Responsible for destroying a handle, if it's required for some backend.
   * It may be needed for some backends. For example, resources associated with
   * this handle needs to be released. This method is called when the execution
   * plan is destroyed (i.e., the program is out of its lifespan).
   *
   * @param[in] handle The handle to be destroyed. An opaque handle returned by
   *     `init()`.
   */
  virtual void destroy(__ET_UNUSED DelegateHandle* handle) const {}
};

struct Backend {
  const char* name_;
  PyTorchBackendInterface* interface_ptr_;
};

// The max number of backends that can be registered in
// an app. It's hard coded to 16 because it's not estimated
// to have more than 16 backends in a system. Each table
// element has two pointers, represented by Backend struct.
// The memory overhead for this table is minimum (only a few bytes).
constexpr size_t kRegistrationTableMaxSize = 16;

class BackendRegistry {
 public:
  BackendRegistry() : registrationTableSize_(0) {}

  /**
   * Registers the Backend object (i.e. string name and PyTorchBackendInterface
   * pair) so that it could be called via the name during the runtime.
   * @param[in] backend Backend object of the user-defined backend delegate.
   * @retval Error code representing whether registration was successful.
   */
  __ET_NODISCARD Error register_backend(const Backend& backend);

  /**
   * Returns the corresponding object pointer for a given string name.
   * The mapping is populated using register_backend method.
   *
   * @param[in] name Name of the user-defined backend delegate.
   * @retval Pointer to the appropriate object that implements
   *         PyTorchBackendInterface. Nullptr if it can't find anything
   *         with the given name.
   */
  PyTorchBackendInterface* get_backend_class(const char* name);

 private:
  Backend backend_table_[kRegistrationTableMaxSize];
  size_t registrationTableSize_;
};

/**
 * Returns the corresponding object pointer for a given string name.
 * The mapping is populated using register_backend method.
 *
 * @param[in] name Name of the user-defined backend delegate.
 * @retval Pointer to the appropriate object that implements
 *         PyTorchBackendInterface. Nullptr if it can't find anything
 *         with the given name.
 */
PyTorchBackendInterface* get_backend_class(const char* name);

/**
 * Registers the Backend object (i.e. string name and PyTorchBackendInterface
 * pair) so that it could be called via the name during the runtime.
 *
 * @param[in] backend Backend object
 * @retval Error code representing whether registration was successful.
 */
__ET_NODISCARD Error register_backend(const Backend& backend);

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendRegistry;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::get_backend_class;
// using ::executorch::runtime::kRegistrationTableMaxSize;
using ::executorch::runtime::PyTorchBackendInterface;
using ::executorch::runtime::register_backend;
using ::executorch::runtime::SizedBuffer;
} // namespace executor
} // namespace torch
