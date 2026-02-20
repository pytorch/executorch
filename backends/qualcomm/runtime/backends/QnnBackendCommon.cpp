/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

QnnBackend::~QnnBackend() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (nullptr != handle_) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn backend");
    error = qnn_interface.qnn_backend_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "backend_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

void QnnBackend::BackendRegisterOpPackage(
    const flatbuffers::Vector<
        flatbuffers::Offset<qnn_delegate::QnnExecuTorchOpPackageInfo>>*
        op_packages_infos) {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  QnnExecuTorchOpPackagePlatform current_platform =
      QnnExecuTorchOpPackagePlatform::UNKNOWN;
#if defined(__x86_64__)
  current_platform = QnnExecuTorchOpPackagePlatform::X86_64;
#elif defined(__ANDROID__)
  current_platform = QnnExecuTorchOpPackagePlatform::AARCH64_ANDROID;
#endif
  if (current_platform == QnnExecuTorchOpPackagePlatform::UNKNOWN)
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to detect the platform. Only support x86_64 or android.");
  for (const auto op_package_info : *op_packages_infos) {
    if (current_platform != op_package_info->platform() ||
        op_package_manager_.Has(op_package_info->op_package_path()->c_str()))
      continue;

    error = qnn_interface.qnn_backend_register_op_package(
        handle_,
        op_package_info->op_package_path()->c_str(),
        op_package_info->interface_provider()->c_str(),
        EnumNameQnnExecuTorchOpPackageTarget(op_package_info->target()));
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to register op package: "
          "%s , error=%d",
          op_package_info->op_package_path()->c_str(),
          QNN_GET_ERROR_CODE(error));
    } else {
      op_package_manager_.Add(op_package_info->op_package_path()->c_str());
    }
  }
}

Error QnnBackend::Configure(
    const QnnExecuTorchOpPackageOptions* op_package_options) {
  // create qnn backend
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnBackend_Config_t*> temp_backend_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_backend_config) == Error::Ok,
      Internal,
      "Fail to make backend config.");

  error = qnn_interface.qnn_backend_create(
      logger_->GetHandle(),
      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
      &handle_);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to create "
        "backend_handle for Backend "
        "ID %u, error=%d",
        qnn_interface.GetBackendId(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  if (op_package_options->op_package_infos()->size() > 0) {
    BackendRegisterOpPackage(op_package_options->op_package_infos());
  }

  return Error::Ok;
}

Error QnnBackend::VerifyQNNSDKVersion() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();

  Qnn_ApiVersion_t qnn_version = {QNN_VERSION_INIT};
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_backend_get_api_version(&qnn_version);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to get Qnn API version.");
    return Error::Internal;
  }

  Qnn_ApiVersion_t expected_version = {QNN_VERSION_INIT};
  expected_version.coreApiVersion.major = QNN_API_VERSION_MAJOR;
  expected_version.coreApiVersion.minor = QNN_API_VERSION_MINOR;
  expected_version.coreApiVersion.patch = QNN_API_VERSION_PATCH;
  expected_version.backendApiVersion = QNN_VERSION_INIT;
  if (qnn_interface.GetBackendId() == QNN_BACKEND_ID_SAVER) {
    expected_version.backendApiVersion.major = QNN_SAVER_API_VERSION_MAJOR;
    expected_version.backendApiVersion.minor = QNN_SAVER_API_VERSION_MINOR;
    expected_version.backendApiVersion.patch = QNN_SAVER_API_VERSION_PATCH;
  } else {
    expected_version.backendApiVersion = GetExpectedBackendVersion();
  }
  const char* backend_type = EnumNameQnnExecuTorchBackendType(
      static_cast<QnnExecuTorchBackendType>(qnn_interface.GetBackendId()));

  Error status = VersionChecker(
      qnn_version.coreApiVersion, expected_version.coreApiVersion, "Qnn API");
  if (status == Error::Ok) {
    status = VersionChecker(
        qnn_version.backendApiVersion,
        expected_version.backendApiVersion,
        backend_type);
  }

  return status;
}

Error QnnBackend::VersionChecker(
    const Qnn_Version_t& qnn_version,
    const Qnn_Version_t& expected,
    const std::string& prefix) {
  if (qnn_version.major != expected.major) {
    QNN_EXECUTORCH_LOG_ERROR(
        "%s version %u.%u.%u is not supported. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct backend library version.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
    return Error::Internal;
  }
  if (qnn_version.major == QNN_API_VERSION_MAJOR &&
      qnn_version.minor < expected.minor) {
    QNN_EXECUTORCH_LOG_WARN(
        "%s version %u.%u.%u is mismatched. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct backend library version.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
  }
  if ((qnn_version.major == QNN_API_VERSION_MAJOR &&
       qnn_version.minor > expected.minor)) {
    QNN_EXECUTORCH_LOG_WARN(
        "%s version %u.%u.%u is used. "
        "The version is tested against %u.%u.%u.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
