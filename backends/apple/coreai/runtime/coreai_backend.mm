/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "ETCoreAIBridge.h"
#include "coreai_backend_id.h"
#import "coreai_cache_root.h"
#import "coreai_error.h"
#import "coreai_load_coordinator.h"
#import "coreai_manifest.h"
#import "coreai_tensor_io.h"

#include <executorch/runtime/backend/interface.h>
#include <memory>
#include <new>

namespace executorch::backends::coreai {
namespace {
using namespace executorch::runtime;

struct Handle final {
  size_t input_count = 0;
  size_t output_count = 0;
  id<ETCoreAISession> session = nil;
};

Result<NSString*> cache_directory_option(BackendInitContext& context) {
  auto option = context.get_runtime_spec<const char*>("coreai_cache_dir");
  if (!option.ok()) {
    if (option.error() == Error::NotFound) {
      return cache_directory_path(nullptr);
    }
    return option.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      option.get() != nullptr,
      InvalidArgument,
      "Core AI cache directory option must not be null");
  return cache_directory_path(option.get());
}

Result<id<ETCoreAISession>> bind_session(
    id<ETCoreAIPreparedModel> model,
    const SelectedBundle& bundle) {
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block id<ETCoreAISession> session = nil;
  __block NSError* error = nil;
  [model loadFunctionNamed:bundle.function
                inputNames:bundle.inputs
               outputNames:bundle.outputs
                completion:^(id<ETCoreAISession> result, NSError* failure) {
                  session = result;
                  error = failure;
                  dispatch_semaphore_signal(ready);
                }];
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  id<ETCoreAISession> bound = session;
  NSError* failure = error;
  // The SDK may retain the completion block after invoking it.
  session = nil;
  error = nil;
  if (failure != nil) {
    return execution_error(failure);
  }
  ET_CHECK_OR_RETURN_ERROR(
      bound != nil, Internal, "Core AI returned no session");
  return bound;
}

class CoreAIBackend final : public BackendInterface {
 public:
  bool is_available() const override {
    return ETCoreAIIsAvailable();
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    @autoreleasepool {
      ET_CHECK_OR_RETURN_ERROR(
          processed && processed->data() && processed->size() > 0,
          InvalidProgram,
          "Missing Core AI manifest");
      auto manifest = parse_manifest(processed->data(), processed->size());
      if (!manifest.ok()) {
        return manifest.error();
      }
      auto assets = select_host_bundle(manifest.get());
      if (!assets.ok()) {
        return assets.error();
      }
      auto configured = cache_directory_option(context);
      if (!configured.ok()) {
        return configured.error();
      }
      id<ETCoreAIModelLoading> loader = ETCoreAICreateModelLoader();
      ET_CHECK_OR_RETURN_ERROR(
          loader != nil, Internal, "Cannot create Core AI loader");
      auto cache_directory = resolve_cache_directory(
          configured.get(), StorageCreationMode::Create);
      if (!cache_directory.ok()) {
        return cache_directory.error();
      }
      ET_CHECK_OR_RETURN_ERROR(
          cache_directory->has_value(),
          AccessFailed,
          "Core AI cache directory disappeared");
      auto prepared = acquire_model(
          assets.get(),
          context.get_named_data_map(),
          cache_directory->value(),
          loader);
      if (!prepared.ok()) {
        return prepared.error();
      }
      auto session = bind_session(prepared.get(), assets->bundle);
      if (!session.ok()) {
        return session.error();
      }
      auto handle = std::unique_ptr<Handle>(new (std::nothrow) Handle());
      ET_CHECK_OR_RETURN_ERROR(
          handle != nullptr,
          MemoryAllocationFailed,
          "Cannot allocate Core AI handle");
      handle->input_count = assets->bundle.inputs.count;
      handle->output_count = assets->bundle.outputs.count;
      handle->session = session.get();
      processed->Free();
      return static_cast<DelegateHandle*>(handle.release());
    }
  }

  Error execute(
      BackendExecutionContext&,
      DelegateHandle* opaque,
      Span<EValue*> args) const override {
    @autoreleasepool {
      ET_CHECK_OR_RETURN_ERROR(
          opaque != nullptr, DelegateInvalidHandle, "Null Core AI handle");
      auto& handle = *static_cast<Handle*>(opaque);
      return execute_session(
          handle.session, args, handle.input_count, handle.output_count);
    }
  }

  void destroy(DelegateHandle* handle) const override {
    @autoreleasepool {
      delete static_cast<Handle*>(handle);
    }
  }
};

CoreAIBackend backend;
[[maybe_unused]] const auto registration =
    register_backend({kCoreAIBackendId, &backend});
} // namespace
} // namespace executorch::backends::coreai
