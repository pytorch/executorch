/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>

#include <array>
#include <string>
#include <unordered_set>
namespace torch {
namespace executor {
//  ================== Util function defs ===========================
enum QnnExecuTorchOptionsTypes {
  kUndefinedOption = 0,
  kBackendType,
  kLibraryPath,
  kSkelLibraryDir,
  kLogLevel,
  kHtpSocModel,
  kHtpPerformanceMode,
  kHtpPrecision,
  kHtpPdSession,
  kHtpUseConvHmx,
  kHtpUseFoldRelu,
  kNumOptions
};

constexpr std::array<std::pair<const char*, int>, kNumOptions - 1>
    kOptionsMap // NOLINT(cert-err58-cpp)
    = {{
        {"backend_type", kBackendType},
        {"library_path", kLibraryPath},
        {"skel_library_dir", kSkelLibraryDir},
        {"log_level", kLogLevel},
        {"htp_soc_model", kHtpSocModel},
        {"htp_performance_mode", kHtpPerformanceMode},
        {"htp_precision", kHtpPrecision},
        {"htp_pd_session", kHtpPdSession},
        {"htp_use_conv_hmx", kHtpUseConvHmx},
        {"htp_use_fold_relu", kHtpUseFoldRelu},
    }};

template <std::size_t SIZE>
int FindOptionInMap(
    const char* option,
    const std::array<std::pair<const char*, int>, SIZE> map) {
  for (size_t i = 0; i < map.size(); ++i) {
    if (std::strcmp(option, (map.at(i)).first) == 0) {
      return (map.at(i)).second;
    }
  }
  return kUndefinedOption;
}

// ========== Public method implementations =========================
using namespace qnn;

Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  QnnExecuTorchOptions options = QnnExecuTorchOptionsDefault();

  // covert SizedBuffer to qnn ExecuTorch option
  options.qnn_context_blob.buffer = const_cast<void*>(processed->data());
  options.qnn_context_blob.nbytes = processed->size();

  // covert CompileSpec to qnn ExecuTorch option
  for (auto& compile_spec : compile_specs) {
    auto type = static_cast<QnnExecuTorchOptionsTypes>(
        FindOptionInMap(compile_spec.key, kOptionsMap));

    switch (type) {
      case kBackendType:
        options.backend_type = *static_cast<const QnnExecuTorchBackendType*>(
            compile_spec.value.buffer);
        break;
      case kLogLevel:
        options.log_level = *static_cast<const QnnExecuTorchLogLevel*>(
            compile_spec.value.buffer);
        break;
      case kLibraryPath:
        options.library_path =
            static_cast<const char*>(compile_spec.value.buffer);
        break;
      case kSkelLibraryDir:
        options.skel_library_dir =
            static_cast<const char*>(compile_spec.value.buffer);
        break;
      case kHtpSocModel:
        options.htp_options.soc_model =
            *static_cast<const QcomChipset*>(compile_spec.value.buffer);
        break;
      case kHtpPerformanceMode:
        options.htp_options.performance_mode =
            *static_cast<const QnnExecuTorchHtpPerformanceMode*>(
                compile_spec.value.buffer);
        break;
      case kHtpUseConvHmx:
        options.htp_options.use_conv_hmx =
            *static_cast<const bool*>(compile_spec.value.buffer);
        break;
      case kHtpUseFoldRelu:
        options.htp_options.use_fold_relu =
            *static_cast<const bool*>(compile_spec.value.buffer);
        break;
      case kHtpPrecision:
        options.htp_options.precision =
            *static_cast<const QnnExecuTorchHtpPrecision*>(
                compile_spec.value.buffer);
        break;
      case kHtpPdSession:
        options.htp_options.pd_session =
            *static_cast<const QnnExecuTorchHtpPdSession*>(
                compile_spec.value.buffer);
        break;
      case kUndefinedOption:
      default:
        QNN_EXECUTORCH_LOG(
            kLogLevelWarn,
            "[Qnn ExecuTorch]: unknown argument: %s",
            compile_spec.key);
    }
  }
  // Create QnnManager
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  QnnManager* qnn_manager =
      ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, QnnManager);

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(&options);

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Init() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->AllocateTensor() == Error::Ok,
      Internal,
      "Fail to allocate tensor");
  return qnn_manager;
}

Error QnnExecuTorchBackend::execute(
    __ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    EValue** args) const {
  QnnManager* qnn_manager = static_cast<QnnManager*>(handle);

  std::vector<std::shared_ptr<TensorWrapper>> input_tensors =
      qnn_manager->GetGraphInputs();
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors =
      qnn_manager->GetGraphOutputs();
  std::vector<Qnn_Tensor_t> input_tensor_structs;
  std::vector<Qnn_Tensor_t> output_tensor_structs;

  for (int i = 0; i < input_tensors.size(); ++i) {
    input_tensors[i]->FillDataBuffer(
        args[i]->toTensor().const_data_ptr(), true /* copy_data */);
    input_tensor_structs.push_back(input_tensors[i]->CloneTensorStruct());
  }

  for (int i = input_tensors.size();
       i < input_tensors.size() + output_tensors.size();
       ++i) {
    output_tensors[i - input_tensors.size()]->FillDataBuffer(
        args[i]->toTensor().mutable_data_ptr(), false /* copy_data */);
    output_tensor_structs.push_back(
        output_tensors[i - input_tensors.size()]->CloneTensorStruct());
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Execute(input_tensor_structs, output_tensor_structs) ==
          Error::Ok,
      Internal,
      "Fail to execute graph");
  return Error::Ok;
}

void QnnExecuTorchBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    QnnManager* qnn_manager = static_cast<QnnManager*>(handle);
    qnn_manager->Destroy();
  }
}

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

namespace {
auto cls = QnnExecuTorchBackend();
Backend backend{"QnnBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
} // namespace executor
} // namespace torch
