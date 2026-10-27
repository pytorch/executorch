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

Result<NSString*> path_option(BackendInitContext& context, const char* key) {
  auto option = context.get_runtime_spec<const char*>(key);
  if (!option.ok()) {
    if (option.error() == Error::NotFound) {
      return static_cast<NSString*>(nil);
    }
    return option.error();
  }
  NSString* path = option.get() == nullptr
      ? nil
      : [NSString stringWithUTF8String:option.get()];
  ET_CHECK_OR_RETURN_ERROR(
      path != nil && path.isAbsolutePath,
      InvalidArgument,
      "Core AI %s must be an absolute UTF-8 path",
      key);
  return path;
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
      auto assets = select_runtime_assets(manifest.get());
      if (!assets.ok()) {
        return assets.error();
      }
      auto assets_option = path_option(context, "coreai_assets_dir");
      if (!assets_option.ok()) {
        return assets_option.error();
      }
      NSString* root = assets_option.get();
      if (root == nil) {
        auto default_root = default_coreai_assets_root();
        if (!default_root.ok()) {
          return default_root.error();
        }
        root = default_root.get();
      }
      id<ETCoreAIModelLoading> loader = ETCoreAICreateModelLoader();
      ET_CHECK_OR_RETURN_ERROR(
          loader != nil, Internal, "Cannot create Core AI loader");
      auto prepared = acquire_bookmark_model(
          assets.get(), context.get_named_data_map(), root, loader);
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
