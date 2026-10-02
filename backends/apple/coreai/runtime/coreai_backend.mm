/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ETCoreAIBridge.h"
#import "coreai_assets.h"
#import "coreai_load_coordinator.h"
#import "coreai_storage.h"

#include <TargetConditionals.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace executorch::backends::coreai {
namespace {
using namespace executorch::runtime;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;

struct Handle {
  size_t input_count = 0;
  size_t output_count = 0;
  id<ETCoreAISession> session = nil;
};

Error bridge_error(NSError* error) {
  ET_LOG(Error, "Core AI: %s", error.localizedDescription.UTF8String);
  if ([error.domain isEqualToString:ETCoreAIErrorDomain]) {
    switch (error.code) {
      case ETCoreAIErrorInvalidModel:
        return Error::InvalidProgram;
      case ETCoreAIErrorUnsupported:
        return Error::NotSupported;
      case ETCoreAIErrorInvalidArgument:
        return Error::InvalidArgument;
      default:
        break;
    }
  }
  return Error::Internal;
}

Result<ETCoreAIScalarType> scalar_type(ScalarType type) {
  switch (type) {
    case ScalarType::Half:
      return ETCoreAIScalarTypeFloat16;
    case ScalarType::Float:
      return ETCoreAIScalarTypeFloat32;
    default:
      ET_LOG(Error, "Core AI runtime supports FP16 and FP32 tensors only");
      return Error::NotSupported;
  }
}

Result<size_t> tensor_bytes(const Tensor& tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      tensor.device_type() == executorch::aten::DeviceType::CPU,
      NotSupported,
      "Core AI requires CPU-accessible tensor storage");
  ET_CHECK_OR_RETURN_ERROR(
      tensor.dim() >= 0 &&
          static_cast<size_t>(tensor.dim()) <= kTensorDimensionLimit,
      InvalidArgument,
      "Invalid Core AI tensor rank");
  ET_CHECK_OR_RETURN_ERROR(
      tensor_is_default_dim_order(tensor),
      NotSupported,
      "Core AI requires default tensor dimension order");
  size_t elements = 1;
  size_t expected_stride = 1;
  for (size_t i = tensor.sizes().size(); i > 0; --i) {
    const auto size = tensor.sizes()[i - 1];
    ET_CHECK_OR_RETURN_ERROR(
        size >= 0, InvalidArgument, "Negative Core AI tensor dimension");
    ET_CHECK_OR_RETURN_ERROR(
        tensor.strides()[i - 1] >= 0 &&
            static_cast<size_t>(tensor.strides()[i - 1]) == expected_stride,
        NotSupported,
        "Core AI requires contiguous tensor storage");
    ET_CHECK_OR_RETURN_ERROR(
        !c10::mul_overflows(elements, static_cast<size_t>(size), &elements) &&
            !c10::mul_overflows(
                expected_stride,
                std::max(static_cast<size_t>(size), size_t(1)),
                &expected_stride),
        InvalidArgument,
        "Core AI tensor size overflow");
  }
  size_t bytes = 0;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::mul_overflows(
          elements, static_cast<size_t>(tensor.element_size()), &bytes) &&
          bytes <= static_cast<size_t>(std::numeric_limits<NSInteger>::max()) &&
          bytes == tensor.nbytes(),
      InvalidArgument,
      "Invalid Core AI tensor byte count");
  return bytes;
}

Error validate_arguments(
    Span<EValue*> args,
    size_t input_count,
    size_t output_count) {
  ET_CHECK_OR_RETURN_ERROR(
      args.size() >= input_count &&
          args.size() - input_count == output_count,
      InvalidArgument,
      "Core AI argument count mismatch");
  for (EValue* arg : args) {
    ET_CHECK_OR_RETURN_ERROR(
        arg && arg->isTensor(),
        InvalidArgument,
        "Core AI arguments must be tensors");
    auto type = scalar_type(arg->toTensor().scalar_type());
    if (!type.ok()) {
      return type.error();
    }
    auto bytes = tensor_bytes(arg->toTensor());
    if (!bytes.ok()) {
      return bytes.error();
    }
  }
  return Error::Ok;
}

Result<NSArray<ETCoreAIInputTensor*>*> borrow_inputs(
    Span<EValue*> args,
    size_t input_count) {
  NSMutableArray<ETCoreAIInputTensor*>* inputs =
      [NSMutableArray arrayWithCapacity:input_count];
  for (size_t i = 0; i < input_count; ++i) {
    const auto& tensor = args[i]->toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        tensor.nbytes() == 0 || tensor.const_data_ptr() != nullptr,
        InvalidArgument,
        "Core AI input has no storage");
    NSMutableArray<NSNumber*>* shape =
        [NSMutableArray arrayWithCapacity:tensor.dim()];
    for (auto size : tensor.sizes()) {
      [shape addObject:@(size)];
    }
    [inputs addObject:[[ETCoreAIInputTensor alloc]
                          initWithBytes:tensor.const_data_ptr()
                              byteCount:tensor.nbytes()
                                  shape:shape
                             scalarType:scalar_type(tensor.scalar_type())
                                            .get()]];
  }
  return inputs;
}

Error copy_outputs(
    NSArray<ETCoreAITensor*>* outputs,
    Span<EValue*> args,
    size_t input_count,
    size_t output_count) {
  ET_CHECK_OR_RETURN_ERROR(
      outputs != nil && outputs.count == output_count,
      InvalidExternalData,
      "Core AI output count mismatch");
  std::vector<std::vector<SizesType>> shapes(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    ETCoreAITensor* output = outputs[i];
    const auto& tensor = args[input_count + i]->toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        [output isKindOfClass:ETCoreAITensor.class] &&
            output.scalarType == scalar_type(tensor.scalar_type()).get() &&
            output.shape.count == static_cast<size_t>(tensor.dim()),
        InvalidExternalData,
        "Core AI output type or rank mismatch");
    size_t bytes = tensor.element_size();
    for (id size in output.shape) {
      ET_CHECK_OR_RETURN_ERROR(
          [size isKindOfClass:NSNumber.class] &&
              [size longLongValue] >= 0 &&
              [size longLongValue] <=
                  std::numeric_limits<SizesType>::max() &&
              [size isEqualToNumber:@([size longLongValue])],
          InvalidExternalData,
          "Invalid Core AI output dimension");
      const auto dimension = static_cast<SizesType>([size longLongValue]);
      shapes[i].push_back(dimension);
      ET_CHECK_OR_RETURN_ERROR(
          !c10::mul_overflows(
              bytes, static_cast<size_t>(dimension), &bytes),
          InvalidExternalData,
          "Core AI output size overflow");
    }
    ET_CHECK_OR_RETURN_ERROR(
        output.data != nil && bytes == output.data.length,
        InvalidExternalData,
        "Core AI output byte count mismatch");
  }
  // Check all resized outputs before copying to avoid partial result writes.
  for (size_t i = 0; i < output_count; ++i) {
    auto& tensor = args[input_count + i]->toTensor();
    ET_CHECK_OK_OR_RETURN_ERROR(resize_tensor(
        tensor, ArrayRef<SizesType>(shapes[i].data(), shapes[i].size())));
    auto bytes = tensor_bytes(tensor);
    if (!bytes.ok()) {
      return bytes.error();
    }
    ET_CHECK_OR_RETURN_ERROR(
        bytes.get() == outputs[i].data.length &&
            (bytes.get() == 0 || tensor.mutable_data_ptr() != nullptr),
        InvalidArgument,
        "Core AI output has insufficient storage");
  }
  for (size_t i = 0; i < output_count; ++i) {
    NSData* data = outputs[i].data;
    if (data.length > 0) {
      memcpy(
          args[input_count + i]->toTensor().mutable_data_ptr(),
          data.bytes,
          data.length);
    }
  }
  return Error::Ok;
}

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

class CoreAIBackend final : public BackendInterface {
 public:
  bool is_available() const override { return ETCoreAIIsAvailable(); }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    @autoreleasepool {
      ET_CHECK_OR_RETURN_ERROR(
          is_available(), NotSupported, "Core AI requires macOS 27 or iOS 27");
      ET_CHECK_OR_RETURN_ERROR(
          processed && processed->data() && processed->size() > 0,
          InvalidProgram,
          "Missing Core AI manifest");
      NSData* bytes =
          [NSData dataWithBytesNoCopy:const_cast<void*>(processed->data())
                               length:processed->size()
                         freeWhenDone:NO];
      auto manifest = parse_manifest(bytes);
      if (!manifest.ok()) {
        return manifest.error();
      }
      ET_CHECK_OR_RETURN_ERROR(
          [NSProcessInfo.processInfo
              isOperatingSystemAtLeastVersion:manifest->minimum_version],
          DelegateInvalidCompatibility,
          "Core AI model requires a newer operating system");
#if TARGET_OS_OSX
      NSString* platform = @"macOS";
#elif TARGET_OS_IOS
      NSString* platform = @"iOS";
#else
      NSString* platform = @"unsupported";
#endif
      NSString* architecture = ETCoreAIDeviceArchitectureName();
      auto selected = select_assets(manifest.get(), architecture, platform);
      if (!selected.ok()) {
        return selected.error();
      }
      manifest.get() = std::move(selected.get());
      auto assets_option = path_option(context, "coreai_assets_dir");
      if (!assets_option.ok()) {
        return assets_option.error();
      }
      NSString* coreai_assets_dir = assets_option.get();
      if (coreai_assets_dir == nil) {
        auto default_root = default_coreai_assets_root();
        if (!default_root.ok()) {
          return default_root.error();
        }
        coreai_assets_dir = default_root.get();
      }
      id<ETCoreAIModelLoading> loader = ETCoreAICreateModelLoader();
      ET_CHECK_OR_RETURN_ERROR(
          loader != nil, Internal, "Cannot create Core AI loader");
      __block id<ETCoreAISession> session = nil;
      __block NSError* load_error = nil;
      dispatch_semaphore_t ready = dispatch_semaphore_create(0);
      auto prepared = acquire_bookmark_model(
          manifest.get(),
          context.get_named_data_map(),
          coreai_assets_dir,
          platform,
          architecture,
          loader);
      if (!prepared.ok()) {
        return prepared.error();
      }
      [prepared.get()
          loadFunctionNamed:manifest->function
                 inputNames:manifest->inputs
                outputNames:manifest->outputs
                 completion:^(id<ETCoreAISession> result, NSError* error) {
                   session = result;
                   load_error = error;
                   dispatch_semaphore_signal(ready);
                 }];
      dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
      if (load_error != nil) {
        session = nil;
        return bridge_error(load_error);
      }
      ET_CHECK_OR_RETURN_ERROR(
          session != nil, Internal, "Core AI returned no session");
      auto handle = std::unique_ptr<Handle>(new (std::nothrow) Handle());
      ET_CHECK_OR_RETURN_ERROR(
          handle != nullptr,
          MemoryAllocationFailed,
          "Cannot allocate Core AI handle");
      handle->input_count = manifest->inputs.count;
      handle->output_count = manifest->outputs.count;
      handle->session = session;
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
      const size_t input_count = handle.input_count;
      const size_t output_count = handle.output_count;
      ET_CHECK_OK_OR_RETURN_ERROR(
          validate_arguments(args, input_count, output_count));
      auto inputs = borrow_inputs(args, input_count);
      if (!inputs.ok()) {
        return inputs.error();
      }
      __block NSArray<ETCoreAITensor*>* outputs = nil;
      __block NSError* execution_error = nil;
      dispatch_semaphore_t ready = dispatch_semaphore_create(0);
      [handle.session
          executeInputs:inputs.get()
             completion:^(NSArray<ETCoreAITensor*>* result, NSError* error) {
               outputs = result;
               execution_error = error;
               dispatch_semaphore_signal(ready);
             }];
      // ET may reuse input storage after execute returns, never before.
      dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
      if (execution_error != nil) {
        return bridge_error(execution_error);
      }
      return copy_outputs(outputs, args, input_count, output_count);
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
    register_backend({"CoreAIBackend", &backend});
}  // namespace
}  // namespace executorch::backends::coreai
