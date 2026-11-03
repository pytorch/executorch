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
#include <cstring>
#include <memory>
#include <new>

namespace executorch::backends::coreai {
namespace {
using namespace executorch::runtime;

struct Handle final {
  size_t input_count = 0;
  size_t output_count = 0;
  ETCoreAILoadedModel* model = nil;
  id<ETCoreAISession> session = nil;
};

template <typename T>
Result<T>
runtime_option(BackendInitContext& context, const char* key, T fallback) {
  auto option = context.get_runtime_spec<T>(key);
  if (!option.ok() && option.error() == Error::NotFound) {
    return fallback;
  }
  return option;
}

Result<NSString*> cache_directory_option(BackendInitContext& context) {
  auto option =
      runtime_option<const char*>(context, "coreai_cache_dir", nullptr);
  if (!option.ok()) {
    return option.error();
  }
  return cache_directory_path(option.get());
}

Result<ETCoreAIComputeUnit> compute_unit_option(BackendInitContext& context) {
  auto option =
      runtime_option<const char*>(context, "coreai_compute_unit", "default");
  if (!option.ok()) {
    return option.error();
  }
  static constexpr struct {
    const char* name;
    ETCoreAIComputeUnit value;
  } kComputeUnits[] = {
      {"default", ETCoreAIComputeUnitDefault},
      {"cpu_only", ETCoreAIComputeUnitCPUOnly},
      {"prefer_cpu", ETCoreAIComputeUnitPreferCPU},
      {"prefer_gpu", ETCoreAIComputeUnitPreferGPU},
      {"prefer_neural_engine", ETCoreAIComputeUnitPreferNeuralEngine},
  };
  for (const auto& unit : kComputeUnits) {
    if (std::strcmp(option.get(), unit.name) == 0) {
      return unit.value;
    }
  }
  ET_LOG(
      Error,
      "Unknown Core AI compute unit '%s'; expected default, cpu_only, "
      "prefer_cpu, prefer_gpu or prefer_neural_engine",
      option.get());
  return Error::InvalidArgument;
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
      auto host_bundle = select_host_bundle(manifest.get());
      if (!host_bundle.ok()) {
        return host_bundle.error();
      }
      auto configured = cache_directory_option(context);
      if (!configured.ok()) {
        return configured.error();
      }
      auto compute_unit = compute_unit_option(context);
      if (!compute_unit.ok()) {
        return compute_unit.error();
      }
      auto share = runtime_option<bool>(
          context, "coreai_share_multifunction_model", false);
      if (!share.ok()) {
        return share.error();
      }
      ET_CHECK_OR_RETURN_ERROR(
          !share.get() || manifest->multifunction,
          InvalidArgument,
          "coreai_share_multifunction_model requires a multi-function export "
          "(CoreAIPartitioner(multifunction=True))");
      id<ETCoreAIModelLoading> loader =
          ETCoreAICreateModelLoader(compute_unit.get());
      ET_CHECK_OR_RETURN_ERROR(
          loader != nil, Internal, "Cannot create Core AI loader");
      auto cache_directory = create_cache_directory(configured.get());
      if (!cache_directory.ok()) {
        return cache_directory.error();
      }
      auto loaded = acquire_loaded_model(
          host_bundle.get(),
          context.get_named_data_map(),
          cache_directory.get(),
          loader,
          share.get());
      if (!loaded.ok()) {
        return loaded.error();
      }
      auto session = bind_session(loaded.get().model, host_bundle->bundle);
      if (!session.ok()) {
        return session.error();
      }
      auto handle = std::unique_ptr<Handle>(new (std::nothrow) Handle());
      ET_CHECK_OR_RETURN_ERROR(
          handle != nullptr,
          MemoryAllocationFailed,
          "Cannot allocate Core AI handle");
      handle->input_count = host_bundle->bundle.inputs.count;
      handle->output_count = host_bundle->bundle.outputs.count;
      handle->model = loaded.get();
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
      std::lock_guard<std::mutex> guard([handle.model executionMutex]);
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
