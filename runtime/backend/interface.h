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

namespace torch {
namespace executor {

struct SizedBuffer {
  void* buffer;
  size_t nbytes; // number of bytes of buffer
};

struct RuntimeInfo {
  const char* key; // runtime info key like "runtime_version",
  SizedBuffer value; // runtime info value like "v0.4.2",
};

struct CompileSpec {
  const char* key; // spec key
  SizedBuffer value; // spec value
};

/*

The following is the interface for backend delegation on runtime side, which is
responsible for taking the preprocessed delegated payload from AOT
(ahead-of-time), further compiler and execution.

Compatibility:
Since each backend will have their own binary (the delegated payload) and
runtime, and it's possible that the binary is not compatible with the current
runtime, as known as the BC/FC compatibility. To allow the backend to run
compatibility check for the binary against the backend runtime, here is the
interface:

AOT (Ahead-of-time):
1. Implement the backend_details::is_compatible that takes the preprocessed
binary, the serialized runtime info (from the runtime api) and the compile spec,
which will be invoked by the high level is_compatible API from ExecuTorch.


Runtime:
1. There will be no is_compatible API in runtime. The expectation is that during
init, if the binary is not compatible, return the error code
Error::DelegateInvalidCompatibility

2. Implement the PyTorchBackendInterface::runtime_info function, the serialized
runtime info will be used together with backend_details::is_compatible such that
the backend can compare the binary against the runtime.
*/

/**
 * An opaque handle managed by a backend. Typically points to a backend-private
 * class/struct.
 */
using DelegateHandle = void;

class PyTorchBackendInterface {
 public:
  virtual ~PyTorchBackendInterface() = 0;

  /**
   * Returns the runtime info from the backend. The key is the like
   * runtime_version, supported binary version, available library, etc.
   * The value is SizedBuffer which can be used to cover more data type,
   * including int, float, str, etc. The runtime info is a serializable format
   * and it can be used AOT to figure out whether the preprocessed bytes (the
   * delegated payload) is compatible with the current backend.
   */
  // TODO: change this class to be virtual after we add the implementaion for
  // each backend.
  virtual Error runtime_info(
      __ET_UNUSED ArrayRef<RuntimeInfo> runtime_info) const {
    return Error::Ok;
  }

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

} // namespace executor
} // namespace torch
