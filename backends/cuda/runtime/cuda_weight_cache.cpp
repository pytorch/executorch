/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_weight_cache.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using aoti::AOTInductorConstantMapEntry;
using aoti::AtenTensorHandle;
using aoti::slim::SlimTensor;
using aoti::slim::c10::Device;
using runtime::Error;
using runtime::NamedDataMap;

Error CudaWeightCache::validate_view(const CudaFqnWeightEntry& entry) {
  uint64_t item_size = 0;
  switch (static_cast<aoti::slim::c10::ScalarType>(entry.dtype)) {
    case aoti::slim::c10::ScalarType::Byte:
    case aoti::slim::c10::ScalarType::Char:
    case aoti::slim::c10::ScalarType::Bool:
      item_size = 1;
      break;
    case aoti::slim::c10::ScalarType::Short:
    case aoti::slim::c10::ScalarType::Half:
    case aoti::slim::c10::ScalarType::BFloat16:
      item_size = 2;
      break;
    case aoti::slim::c10::ScalarType::Int:
    case aoti::slim::c10::ScalarType::Float:
      item_size = 4;
      break;
    case aoti::slim::c10::ScalarType::Long:
      item_size = 8;
      break;
    default:
      return Error::InvalidProgram;
  }

  ET_CHECK_OR_RETURN_ERROR(
      entry.storage_nbytes <= std::numeric_limits<size_t>::max(),
      InvalidProgram,
      "CUDA FQN storage '%s' is too large for this platform",
      entry.storage_key.c_str());

  bool empty = false;
  uint64_t last_element = static_cast<uint64_t>(entry.storage_offset);
  for (size_t dim = 0; dim < entry.sizes.size(); ++dim) {
    const uint64_t size = static_cast<uint64_t>(entry.sizes[dim]);
    const uint64_t stride = static_cast<uint64_t>(entry.strides[dim]);
    if (size == 0) {
      empty = true;
      break;
    }
    const uint64_t extent = size - 1;
    ET_CHECK_OR_RETURN_ERROR(
        extent == 0 || stride <= std::numeric_limits<uint64_t>::max() / extent,
        InvalidProgram,
        "CUDA FQN weight '%s' has overflowing shape/stride metadata",
        entry.fqn.c_str());
    const uint64_t span = stride * extent;
    ET_CHECK_OR_RETURN_ERROR(
        last_element <= std::numeric_limits<uint64_t>::max() - span,
        InvalidProgram,
        "CUDA FQN weight '%s' has overflowing storage metadata",
        entry.fqn.c_str());
    last_element += span;
  }

  uint64_t required_nbytes = 0;
  if (!empty) {
    ET_CHECK_OR_RETURN_ERROR(
        last_element < std::numeric_limits<uint64_t>::max() &&
            last_element + 1 <=
                std::numeric_limits<uint64_t>::max() / item_size,
        InvalidProgram,
        "CUDA FQN weight '%s' has overflowing storage size",
        entry.fqn.c_str());
    required_nbytes = (last_element + 1) * item_size;
  }
  ET_CHECK_OR_RETURN_ERROR(
      required_nbytes <= entry.storage_nbytes,
      InvalidProgram,
      "CUDA FQN weight '%s' requires %llu bytes from a %llu-byte storage",
      entry.fqn.c_str(),
      static_cast<unsigned long long>(required_nbytes),
      static_cast<unsigned long long>(entry.storage_nbytes));
  return Error::Ok;
}

Error CudaWeightCache::acquire_storage(
    const NamedDataMap* named_data_map,
    const CudaFqnWeightEntry& entry,
    uintptr_t logical_scope,
    int device_index,
    std::shared_ptr<CudaWeightStorage>& storage,
    bool& reused) const {
  reused = false;
  ET_CHECK_OR_RETURN_ERROR(
      named_data_map != nullptr,
      InvalidArgument,
      "CUDA FQN weights require a named data map");

  const auto device_type =
      static_cast<aoti::slim::c10::DeviceType>(entry.device_type);
  const bool is_cuda_storage = device_type == aoti::slim::c10::DeviceType::CUDA;
  const int storage_device_index = is_cuda_storage ? device_index : 0;
  const std::string cache_key = entry.storage_key + "@" +
      std::to_string(logical_scope) +
      (is_cuda_storage ? "@cuda:" + std::to_string(device_index) : "@cpu");

  std::unique_lock<std::mutex> lock(mutex_);
  auto cached = storages_.find(cache_key);
  if (cached != storages_.end()) {
    storage = cached->second.lock();
  }
  if (storage != nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        storage->nbytes == entry.storage_nbytes &&
            storage->device_type == device_type &&
            storage->device_index == storage_device_index,
        InvalidProgram,
        "CUDA FQN weight '%s' has inconsistent allocation metadata",
        entry.fqn.c_str());
    reused = true;
    return Error::Ok;
  }

  auto host_data = named_data_map->get_data(entry.storage_key.c_str());
  ET_CHECK_OR_RETURN_ERROR(
      host_data.ok(),
      NotFound,
      "CUDA FQN storage '%s' is missing from named data",
      entry.storage_key.c_str());
  if (host_data->size() != entry.storage_nbytes) {
    const size_t actual_size = host_data->size();
    host_data->Free();
    ET_LOG(
        Error,
        "CUDA FQN storage '%s' has size %zu, expected %llu",
        entry.storage_key.c_str(),
        actual_size,
        static_cast<unsigned long long>(entry.storage_nbytes));
    return Error::InvalidProgram;
  }

  void* storage_data = nullptr;
  const size_t allocation_size =
      std::max<size_t>(1, static_cast<size_t>(entry.storage_nbytes));
  if (is_cuda_storage) {
    const cudaError_t allocation_error =
        cudaMalloc(&storage_data, allocation_size);
    if (allocation_error != cudaSuccess) {
      host_data->Free();
      ET_LOG(
          Error,
          "cudaMalloc failed for FQN storage '%s': %s",
          entry.storage_key.c_str(),
          cudaGetErrorString(allocation_error));
      return Error::MemoryAllocationFailed;
    }
  } else {
    storage_data = std::malloc(allocation_size);
    if (storage_data == nullptr) {
      host_data->Free();
      ET_LOG(
          Error,
          "malloc failed for CPU FQN storage '%s'",
          entry.storage_key.c_str());
      return Error::MemoryAllocationFailed;
    }
  }

  const auto free_storage_data = [&]() {
    if (is_cuda_storage) {
      (void)cudaFree(storage_data);
    } else {
      std::free(storage_data);
    }
  };

  cudaError_t copy_error = cudaSuccess;
  if (entry.storage_nbytes > 0) {
    if (is_cuda_storage) {
      copy_error = cudaMemcpy(
          storage_data,
          host_data->data(),
          static_cast<size_t>(entry.storage_nbytes),
          cudaMemcpyHostToDevice);
    } else {
      std::memcpy(
          storage_data,
          host_data->data(),
          static_cast<size_t>(entry.storage_nbytes));
    }
  }
  host_data->Free();
  if (copy_error != cudaSuccess) {
    free_storage_data();
    ET_LOG(
        Error,
        "cudaMemcpy failed for FQN storage '%s': %s",
        entry.storage_key.c_str(),
        cudaGetErrorString(copy_error));
    return Error::Internal;
  }

  storage = std::make_shared<CudaWeightStorage>(
      storage_data,
      static_cast<size_t>(entry.storage_nbytes),
      device_type,
      storage_device_index);
  storages_[cache_key] = storage;
  return Error::Ok;
}

Error CudaWeightCache::load(
    CudaDelegateHandle* handle,
    const NamedDataMap* named_data_map,
    const CudaFqnWeightManifest& manifest) const {
  ET_CHECK_OR_RETURN_ERROR(
      named_data_map != nullptr,
      InvalidArgument,
      "CUDA FQN weights require a named data map");
  ET_CHECK_OR_RETURN_ERROR(
      handle->get_num_constants && handle->get_constant_name &&
          handle->get_constant_original_fqn && handle->get_constant_dtype &&
          handle->update_user_managed_constant_buffer_pairs,
      NotSupported,
      "AOTI library does not expose the APIs required by CUDA FQN weights");

  size_t num_constants = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      handle->get_num_constants(handle->container_handle, &num_constants),
      "Failed to enumerate CUDA AOTI constants");
  std::unordered_map<std::string, std::vector<std::string>>
      fqn_to_internal_names;
  std::unordered_map<std::string, int32_t> fqn_to_aoti_dtype;
  for (size_t index = 0; index < num_constants; ++index) {
    const char* internal_name = nullptr;
    const char* fqn = nullptr;
    int32_t dtype = 0;
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->get_constant_name(
            handle->container_handle, index, &internal_name),
        "Failed to read CUDA AOTI constant name at index %zu",
        index);
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->get_constant_original_fqn(
            handle->container_handle, index, &fqn),
        "Failed to read CUDA AOTI constant FQN at index %zu",
        index);
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->get_constant_dtype(handle->container_handle, index, &dtype),
        "Failed to read CUDA AOTI constant dtype at index %zu",
        index);
    if (internal_name != nullptr && fqn != nullptr && fqn[0] != '\0') {
      fqn_to_internal_names[fqn].emplace_back(internal_name);
      auto [metadata, inserted] = fqn_to_aoti_dtype.emplace(fqn, dtype);
      ET_CHECK_OR_RETURN_ERROR(
          inserted || metadata->second == dtype,
          InvalidProgram,
          "CUDA AOTI constant FQN '%s' has inconsistent metadata",
          fqn);
    }
  }

  int device_index = 0;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetDevice(&device_index));
  // Each method receives a different MergedDataMap wrapper, but get_key()
  // forwards the stable key owned by the shared PTD map. Use that identity to
  // scope the process-wide FQN cache to one loaded model.
  std::unordered_map<std::string, uintptr_t> key_scopes;
  auto num_keys = named_data_map->get_num_keys();
  ET_CHECK_OR_RETURN_ERROR(
      num_keys.ok(),
      InvalidProgram,
      "Failed to enumerate CUDA named data while loading FQN weights");
  key_scopes.reserve(num_keys.get());
  for (uint32_t index = 0; index < num_keys.get(); ++index) {
    auto key = named_data_map->get_key(index);
    ET_CHECK_OR_RETURN_ERROR(
        key.ok() && key.get() != nullptr,
        InvalidProgram,
        "Failed to read CUDA named data key %u",
        index);
    key_scopes.emplace(key.get(), reinterpret_cast<uintptr_t>(key.get()));
  }
  std::vector<AOTInductorConstantMapEntry> pairs;
  pairs.reserve(manifest.entries.size());
  std::unordered_set<std::string> bound_fqns;
  size_t reused_storages = 0;
  handle->fqn_weight_tensors.reserve(manifest.entries.size());

  for (const CudaFqnWeightEntry& entry : manifest.entries) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        validate_view(entry), "Invalid CUDA FQN view '%s'", entry.fqn.c_str());
    auto internal_names = fqn_to_internal_names.find(entry.fqn);
    ET_CHECK_OR_RETURN_ERROR(
        internal_names != fqn_to_internal_names.end(),
        InvalidProgram,
        "CUDA FQN weight '%s' is not present in its AOTI library",
        entry.fqn.c_str());
    const auto aoti_dtype = fqn_to_aoti_dtype.find(entry.fqn);
    ET_CHECK_OR_RETURN_ERROR(
        aoti_dtype != fqn_to_aoti_dtype.end() &&
            aoti_dtype->second == entry.dtype,
        InvalidProgram,
        "CUDA FQN weight '%s' dtype does not match its AOTI library "
        "(manifest=%d, AOTI=%d)",
        entry.fqn.c_str(),
        entry.dtype,
        aoti_dtype == fqn_to_aoti_dtype.end() ? -1 : aoti_dtype->second);
    ET_CHECK_OR_RETURN_ERROR(
        bound_fqns.emplace(entry.fqn).second,
        InvalidProgram,
        "CUDA FQN weight '%s' appears more than once in its manifest",
        entry.fqn.c_str());

    std::shared_ptr<CudaWeightStorage> storage;
    bool reused = false;
    const auto logical_scope = key_scopes.find(entry.storage_key);
    ET_CHECK_OR_RETURN_ERROR(
        logical_scope != key_scopes.end(),
        NotFound,
        "CUDA FQN storage '%s' is missing from named data",
        entry.storage_key.c_str());
    ET_CHECK_OK_OR_RETURN_ERROR(
        acquire_storage(
            named_data_map,
            entry,
            logical_scope->second,
            device_index,
            storage,
            reused),
        "Failed to load CUDA FQN storage '%s'",
        entry.storage_key.c_str());
    reused_storages += reused ? 1 : 0;
    handle->fqn_weight_storages.push_back(storage);

    auto tensor = std::make_unique<SlimTensor>(aoti::slim::from_blob(
        storage->data,
        aoti::slim::makeArrayRef(entry.sizes),
        aoti::slim::makeArrayRef(entry.strides),
        static_cast<aoti::slim::c10::ScalarType>(entry.dtype),
        Device(
            static_cast<aoti::slim::c10::DeviceType>(entry.device_type),
            entry.device_type ==
                    static_cast<int32_t>(aoti::slim::c10::DeviceType::CUDA)
                ? device_index
                : 0),
        entry.storage_offset));
    AtenTensorHandle tensor_handle =
        reinterpret_cast<AtenTensorHandle>(tensor.get());
    handle->fqn_weight_tensors.push_back(std::move(tensor));
    for (const std::string& internal_name : internal_names->second) {
      pairs.push_back({internal_name.c_str(), tensor_handle});
    }
  }

  ET_CHECK_OK_OR_RETURN_ERROR(
      handle->update_user_managed_constant_buffer_pairs(
          handle->container_handle,
          pairs.data(),
          pairs.size(),
          /*use_inactive=*/false,
          /*validate_full_update=*/true),
      "Failed to bind CUDA FQN weights");
  ET_LOG(
      Info,
      "Loaded %zu CUDA FQN weights (%zu reused across methods)",
      manifest.entries.size(),
      reused_storages);
  return Error::Ok;
}

} // namespace executorch::backends::cuda
