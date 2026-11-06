/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <dlfcn.h>
#include <sys/system_properties.h>

#include <executorch/backends/samsung/runtime/enn_api_implementation.h>
#include <executorch/backends/samsung/runtime/logging.h>

#include <cstring>
#include <string>
#include <unordered_map>

#define ENN_LOAD_API_FUNC(handle, name, enn_api_ptr)                         \
  enn_api_ptr->name =                                                        \
      reinterpret_cast<name##_fn>(loadApiFunction(handle, #name, false));    \
  if (enn_api_ptr->name == nullptr) {                                        \
    ENN_LOG_ERROR("Unable to access symbols in enn api library: %s", #name); \
    dlclose(handle);                                                         \
    return Error::Internal;                                                  \
  }

namespace torch {
namespace executor {
namespace enn {
void* loadApiFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    ENN_LOG_WARN("Failed to load function %s", name);
  }
  return fn;
}

EnnApi* EnnApi::getEnnApiInstance() {
  static EnnApi enn_api;
  return &enn_api;
}

namespace {

constexpr uint32_t kPerfConfigIdDefault = 0;

struct PerfConfig {
  PerfModePreference mode;
  uint32_t config_id;
};

// Primary source: ro.hardware. Falls back to ro.soc.model if unavailable.
std::string getHwName() {
  char prop_value[PROP_VALUE_MAX] = {};
  if (__system_property_get("ro.hardware", prop_value) > 0) {
    return std::string(prop_value);
  }
  return "";
}

std::string getSocName() {
  char prop_value[PROP_VALUE_MAX] = {};
  if (__system_property_get("ro.soc.model", prop_value) > 0) {
    return std::string(prop_value);
  }
  return "";
}

// Extracts trailing digits, e.g. "exynos9955" -> "9955", "s5e9965" -> "9965".
bool trimToTrailingDigits(std::string& value) {
  const char* digits = "0123456789";
  auto last = value.find_last_of(digits);
  if (last == std::string::npos) {
    return false;
  }
  auto first = last;
  while (first > 0 && std::strchr(digits, value[first - 1]) != nullptr) {
    first--;
  }
  value = value.substr(first, last - first + 1);
  return true;
}

const std::unordered_map<std::string, PerfConfig>& getSocPerfConfigMap() {
  static const std::unordered_map<std::string, PerfConfig> kSocPerfConfig = {
      {"9955", {ENN_PREF_MODE_PERFORMANCE, kPerfConfigIdDefault}},
      {"9965", {ENN_PREF_MODE_CUSTOM, 100111}},
  };
  return kSocPerfConfig;
}

PerfConfig getPerfConfigForSoc() {
  std::string key = getHwName();
  if (key.empty()) {
    key = getSocName();
  }
  if (!key.empty()) {
    trimToTrailingDigits(key);
  }

  const auto& map = getSocPerfConfigMap();
  auto it = map.find(key);
  if (it != map.end()) {
    return it->second;
  }
  return {ENN_PREF_MODE_PERFORMANCE, kPerfConfigIdDefault};
}

// Mirrors LiteRT's litert::samsung::SetGenAiPerfConfigFromSoc: detect the
// SoC from Android system properties and apply its known-good perf mode,
// instead of leaving the ENN runtime on its own default.
void setPerfConfigFromSoc(const EnnApi* enn_api) {
  PerfConfig config = getPerfConfigForSoc();
  if (enn_api->EnnSetPreferencePerfMode(static_cast<uint32_t>(config.mode)) !=
      ENN_RET_SUCCESS) {
    ENN_LOG_ERROR("Failed to set ENN preference perf mode.");
    return;
  }
  if (config.config_id == kPerfConfigIdDefault) {
    return;
  }
  if (enn_api->EnnSetPreferencePerfConfigId == nullptr ||
      enn_api->EnnSetPreferencePerfConfigId(config.config_id) !=
          ENN_RET_SUCCESS) {
    ENN_LOG_WARN(
        "Failed to set ENN perf config id %u, falling back to PERFORMANCE mode.",
        config.config_id);
    enn_api->EnnSetPreferencePerfMode(
        static_cast<uint32_t>(ENN_PREF_MODE_PERFORMANCE));
  }
}

} // namespace

void EnnApi::applyPerfConfigFromSoc() const {
  setPerfConfigFromSoc(this);
}

EnnApi::EnnApi() {
  auto status = loadApiLib();
  if (status != Error::Ok) {
    ET_LOG(Error, "Failed to load enn api library. %s", dlerror());
    return;
  }
  auto ret = EnnInitialize();
  if (ret != ENN_RET_SUCCESS) {
    ET_LOG(Error, "EnnInitialize failed: %d", static_cast<int>(ret));
    unloadApiLib();
    return;
  }
  ET_LOG(Info, "Loading ENN API library Completed.");
  initialize_ = true;
}

EnnApi::~EnnApi() {
  if (initialize_) {
    EnnDeinitialize();
    unloadApiLib();
  }
}

bool EnnApi::isInitialized() const {
  return initialize_;
}

Error EnnApi::loadApiLib() {
  const char enn_api_lib_name[] = "libenn_public_api_cpp.so";
  libenn_public_api_ = dlopen(enn_api_lib_name, RTLD_NOW | RTLD_LOCAL);
  ET_CHECK_OR_RETURN_ERROR(
      libenn_public_api_ != nullptr, Internal, "Lib load failed.");

  ENN_LOAD_API_FUNC(libenn_public_api_, EnnInitialize, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnSetPreferencePerfMode, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnGetPreferencePerfMode, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnOpenModel, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnOpenModelFromMemory, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnSetFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnUnsetFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModelFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModel, this);
  ENN_LOAD_API_FUNC(
      libenn_public_api_, EnnExecuteModelWithSessionIdAsync, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModelWithSessionIdWait, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnCloseModel, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnDeinitialize, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnAllocateAllBuffers, this);
  ENN_LOAD_API_FUNC(
      libenn_public_api_, EnnAllocateAllBuffersWithSessionId, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnBufferCommit, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnGetBuffersInfo, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnReleaseBuffers, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnCreateBuffer, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnReleaseBuffer, this);
  ENN_LOAD_API_FUNC(
      libenn_public_api_, EnnGetFileDescriptorFromEnnBuffer, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnOpenModelFromFd, this);
  // Optional: only newer ENN releases export these.
  EnnSetPreferencePerfConfigId =
      reinterpret_cast<EnnSetPreferencePerfConfigId_fn>(loadApiFunction(
          libenn_public_api_, "EnnSetPreferencePerfConfigId", true));
  EnnOpenModelFromFdWithWeight =
      reinterpret_cast<EnnOpenModelFromFdWithWeight_fn>(loadApiFunction(
          libenn_public_api_, "EnnOpenModelFromFdWithWeight", true));
  return Error::Ok;
}

Error EnnApi::unloadApiLib() {
  if (dlclose(libenn_public_api_) != 0) {
    ET_LOG(Error, "Failed to close ENN API library. %s", dlerror());
    return Error::Internal;
  }
  libenn_public_api_ = nullptr;
  return Error::Ok;
}

} // namespace enn
} // namespace executor
} // namespace torch
