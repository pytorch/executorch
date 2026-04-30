/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI tensor + memory layer impl for the v2 Metal backend.
//
// All buffer/memory work routes through the metal_* C ABI in runtime.h
// so this file stays a .cpp (no Metal/Metal.h required).

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_tensor.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declare validate_dtype (defined in shims/utils.cpp). We don't
// include shims/utils.h here because it pulls in v1's shims/types.h, which
// defines `Tensor = etensor::Tensor` and conflicts with our SlimTensor
// alias from aoti_types.h.
namespace executorch {
namespace backends {
namespace metal {
extern "C" AOTITorchError validate_dtype(int32_t dtype);
} // namespace metal
} // namespace backends
} // namespace executorch

namespace executorch {
namespace backends {
namespace metal {

using namespace executorch::backends::aoti;
namespace slim = executorch::backends::aoti::slim;

extern "C" {

// =====================================================================
// Globals
// =====================================================================

std::unordered_map<Tensor*, std::unique_ptr<Tensor>> tensors;

// Reference counting for memory addresses.
//   NOT_OWN (-1): tensor wraps externally-owned memory; never freed by us.
//   N >= 1     : N live tensor handles share this allocation; the last
//                handle to be deleted frees the underlying buffer.
constexpr int32_t NOT_OWN = -1;
std::unordered_map<void*, int32_t> memory_to_n_tensor;

namespace {

// Convert int64_t sizes/strides arrays into std::vector<int64_t> for use
// with slim::from_blob / IntArrayRef.
std::vector<int64_t> to_int64_vector(int64_t ndim, const int64_t* ptr) {
  if (ptr == nullptr) {
    return {};
  }
  return std::vector<int64_t>(ptr, ptr + ndim);
}

// Compute contiguous (row-major) strides if the caller didn't provide any.
std::vector<int64_t> compute_or_copy_strides(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr) {
  if (strides_ptr != nullptr) {
    return to_int64_vector(ndim, strides_ptr);
  }
  std::vector<int64_t> strides(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; i--) {
      // Match v1 quirk: when next-dim size is 0, just propagate the previous
      // stride rather than zeroing out (avoids degenerate stride patterns).
      strides[i] = (sizes_ptr[i + 1] == 0)
          ? strides[i + 1]
          : strides[i + 1] * sizes_ptr[i + 1];
    }
  }
  return strides;
}

// Insert a SlimTensor into the tensors map and return the raw pointer
// used as the AOTI handle.
Tensor* register_tensor(slim::SlimTensor&& t) {
  auto owned = std::make_unique<slim::SlimTensor>(std::move(t));
  Tensor* raw = owned.get();
  tensors.emplace(raw, std::move(owned));
  return raw;
}

} // namespace

// =====================================================================
// Tensor lifecycle
// =====================================================================

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2[v2]: entered");

  (void)device_type;
  (void)device_index;
  (void)opaque_metadata;
  (void)layout;
  (void)opaque_metadata_size;

  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr, InvalidArgument, "data pointer is null");
  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "sizes_ptr is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr, InvalidArgument, "ret_new_tensor is null");

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Apply storage_offset by adjusting the raw pointer; pass 0 storage_offset
  // to from_blob (mirrors v1 behavior).
  void* adjusted_data = static_cast<char*>(data) +
      (storage_offset * dtype_to_element_size(dtype));

  std::vector<int64_t> sizes = to_int64_vector(ndim, sizes_ptr);
  std::vector<int64_t> strides =
      compute_or_copy_strides(ndim, sizes_ptr, strides_ptr);

  slim::SlimTensor t = slim::from_blob(
      adjusted_data,
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      dtype_to_c10_scalar_type(dtype));

  *ret_new_tensor = register_tensor(std::move(t));

  // Register this address as externally-owned. It must not already be
  // tracked: tensor-from-blob never owns memory it wraps.
  auto memory_it = memory_to_n_tensor.find(adjusted_data);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it == memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is already being tracked by another tensor",
      adjusted_data);
  memory_to_n_tensor[adjusted_data] = NOT_OWN;
  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor) {
  ET_LOG(Debug, "aoti_torch_empty_strided[v2]: entered");
  (void)device_index;

  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  size_t element_size = dtype_to_element_size(dtype);
  ET_CHECK_OR_RETURN_ERROR(
      element_size != 0,
      InvalidArgument,
      "Invalid element size for dtype: %d",
      dtype);
  int64_t nbytes = numel * element_size;

  int32_t mps_device_type = aoti_torch_device_type_mps(); // Returns 13
  if (device_type == mps_device_type) {
    ptr = metal_allocate_buffer(nbytes);
    if (ptr == nullptr) {
      ET_LOG(Error, "Failed to allocate %lld bytes on Metal", nbytes);
      return Error::MemoryAllocationFailed;
    }
  } else if (device_type == 0) { // cpu
    int result = posix_memalign(&ptr, 16, nbytes);
    ET_CHECK_OR_RETURN_ERROR(
        result == 0,
        MemoryAllocationFailed,
        "posix_memalign failed: error %d",
        result);
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr, MemoryAllocationFailed, "posix_memalign returned null");
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        false,
        NotImplemented,
        "empty_strided not implemented for device type %d",
        device_type);
  }

  std::vector<int64_t> sizes = to_int64_vector(ndim, sizes_ptr);
  std::vector<int64_t> strides =
      compute_or_copy_strides(ndim, sizes_ptr, strides_ptr);

  slim::SlimTensor t = slim::from_blob(
      ptr,
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      dtype_to_c10_scalar_type(dtype));

  *ret_new_tensor = register_tensor(std::move(t));

  // This tensor logically owns the buffer (we allocated it). Refcount=1.
  memory_to_n_tensor[ptr] = 1;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  if (tensor == nullptr) {
    return Error::Ok;
  }

  auto it = tensors.find(tensor);
  // Tensors not in the map are temporary views (e.g. CPU ETensor wrappers
  // created by metal_backend_v2 for I/O). Nothing to free.
  if (it == tensors.end()) {
    return Error::Ok;
  }

  void* data_ptr = tensor->data_ptr();
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  if (memory_it != memory_to_n_tensor.end()) {
    int32_t ref_count = memory_it->second;

    if (ref_count == NOT_OWN) {
      tensors.erase(it);
      return Error::Ok;
    } else if (ref_count == 1) {
      if (metal_is_device_pointer(data_ptr)) {
        metal_deallocate_buffer(data_ptr);
      } else {
        free(data_ptr);
      }
      memory_to_n_tensor.erase(memory_it);
    } else if (ref_count > 1) {
      memory_to_n_tensor[data_ptr] = ref_count - 1;
    } else {
      ET_LOG(Error, "Invalid reference count %d for memory %p", ref_count, data_ptr);
      return Error::Internal;
    }
  }

  tensors.erase(it);
  return Error::Ok;
}

AOTITorchError aoti_torch_copy_(
    AOTITensorHandle self,
    AOTITensorHandle src,
    int32_t non_blocking) {
  (void)non_blocking;

  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr, InvalidArgument, "self tensor is null");
  ET_CHECK_OR_RETURN_ERROR(
      src != nullptr, InvalidArgument, "src tensor is null");

  // Dtype compatibility check (same dtype required, like PyTorch copy_).
  auto self_dtype = self->dtype();
  auto src_dtype = src->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      self_dtype == src_dtype,
      InvalidArgument,
      "dtype mismatch. self=%d, src=%d",
      static_cast<int>(self_dtype),
      static_cast<int>(src_dtype));

  // Numel must match.
  size_t self_numel = self->numel();
  size_t src_numel = src->numel();
  ET_CHECK_OR_RETURN_ERROR(
      self_numel == src_numel,
      InvalidArgument,
      "numel mismatch. self=%zu, src=%zu",
      self_numel,
      src_numel);

  // Device classification via the GPU pointer registry (not SlimTensor's
  // own device tag — v2 SlimTensors are all CPU-tagged regardless of
  // whether the buffer is GPU-accessible).
  bool srcIsDevice = metal_is_device_pointer(src->data_ptr());
  bool dstIsDevice = metal_is_device_pointer(self->data_ptr());

  // Same-schema fast path. (TODO: catch (4,1,5) -> (4,5)-style cases.)
  bool same_schema =
      self->dim() == src->dim() && self->dtype() == src->dtype();
  if (same_schema) {
    auto self_strides = self->strides();
    auto src_strides = src->strides();
    for (size_t i = 0; i < self->dim(); i++) {
      if (self_strides[i] != src_strides[i]) {
        same_schema = false;
        break;
      }
    }
  }

  size_t total_bytes = src->nbytes();
  if (same_schema) {
    int result = metal_copy_memory(
        self->data_ptr(),
        src->data_ptr(),
        total_bytes,
        srcIsDevice,
        dstIsDevice);
    ET_CHECK_OR_RETURN_ERROR(
        result == 0, Internal, "metal_copy_memory failed: %d", result);
  } else {
    ET_LOG(Error, "Different schema copies are not implemented yet");
    return Error::NotImplemented;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr, InvalidArgument, "self tensor is null");
  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "sizes_ptr is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr, InvalidArgument, "ret_new_tensor is null");

  int32_t device_type = 0;
  int32_t device_index = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_type(self, &device_type));
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_index(self, &device_index));
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  int32_t dtype = static_cast<int32_t>(self->dtype());
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  void* data_ptr = self->data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked",
      data_ptr);

  void* adjusted_data = static_cast<char*>(data_ptr) +
      (storage_offset * dtype_to_element_size(dtype));

  std::vector<int64_t> sizes = to_int64_vector(ndim, sizes_ptr);
  std::vector<int64_t> strides =
      compute_or_copy_strides(ndim, sizes_ptr, strides_ptr);

  slim::SlimTensor t = slim::from_blob(
      adjusted_data,
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      dtype_to_c10_scalar_type(dtype));

  *ret_new_tensor = register_tensor(std::move(t));

  if (adjusted_data != data_ptr) {
    ET_CHECK_OR_RETURN_ERROR(
        metal_buffer_nocopy(adjusted_data, (*ret_new_tensor)->nbytes(), true),
        Internal,
        "metal_buffer_nocopy failed for adjusted_data %p of size %zu",
        adjusted_data,
        (*ret_new_tensor)->nbytes());
    memory_to_n_tensor[adjusted_data] = NOT_OWN;
  }

  // Bump refcount on the source pointer (only when it's owned, not borrowed).
  if (memory_to_n_tensor[data_ptr] != NOT_OWN) {
    memory_to_n_tensor[data_ptr] += 1;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  ET_CHECK_OR_RETURN_ERROR(
      orig_handle != nullptr, InvalidArgument, "orig_handle is null");
  ET_CHECK_OR_RETURN_ERROR(
      new_handle != nullptr, InvalidArgument, "new_handle is null");

  int32_t device_index = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_device_index(orig_handle, &device_index));
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  int32_t dtype = static_cast<int32_t>(orig_handle->dtype());
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  void* data_ptr = orig_handle->data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked",
      data_ptr);

  // Mirror the original tensor's shape/strides/dtype, sharing storage.
  std::vector<int64_t> sizes(
      orig_handle->sizes().begin(), orig_handle->sizes().end());
  std::vector<int64_t> strides(
      orig_handle->strides().begin(), orig_handle->strides().end());

  slim::SlimTensor t = slim::from_blob(
      data_ptr,
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      orig_handle->dtype());

  *new_handle = register_tensor(std::move(t));

  // Refcount: only bump when the source memory is owned (not borrowed).
  memory_to_n_tensor[data_ptr] = memory_to_n_tensor[data_ptr] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[data_ptr] + 1;
  return Error::Ok;
}

void cleanup_memory() {
  // Use aoti_torch_delete_tensor_object so refcounts/buffer frees stay in
  // sync. Collect keys first since deletion modifies the map.
  std::vector<Tensor*> tensor_ptrs;
  tensor_ptrs.reserve(tensors.size());
  for (const auto& entry : tensors) {
    tensor_ptrs.push_back(entry.first);
  }
  for (Tensor* tensor_ptr : tensor_ptrs) {
    aoti_torch_delete_tensor_object(tensor_ptr);
  }

  tensors.clear();
  metal_cleanup_resources();

  ET_LOG(Info, "[v2] Cleared all tensors and Metal resources");
}

// =====================================================================
// MPS buffer shims
//
// All four route through the metal_* C ABI in runtime.h. This means
// allocations are device-pointer-tracked (so metal_is_device_pointer
// works correctly downstream) and the file stays a .cpp.
// =====================================================================

AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes) {
  if (num_bytes == 0) {
    if (buffer) *buffer = nullptr;
    return Error::Ok;
  }
  if (!buffer) return Error::InvalidArgument;
  *buffer = metal_allocate_buffer(static_cast<long>(num_bytes));
  return *buffer ? Error::Ok : Error::Internal;
}

AOTITorchError aoti_torch_mps_free(void* ptr) {
  if (ptr) metal_deallocate_buffer(ptr);
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start) {
  if (!buffer || !constants_start) return Error::InvalidArgument;

  auto* dst = static_cast<uint8_t*>(buffer) + constant_offset;
  std::memcpy(dst, constants_start + bytes_read, data_size);

  // Register the sub-region so GPU can see it.
  if (constant_offset != 0) {
    metal_buffer_nocopy(dst, data_size, /*map_ptr_to_buffer=*/true);
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset) {
  if (!src_buffer || !dst_buffer) return Error::InvalidArgument;
  // Unified memory — direct memcpy.
  auto* src = static_cast<uint8_t*>(src_buffer) + src_offset;
  auto* dst = static_cast<uint8_t*>(dst_buffer) + dst_offset;
  std::memcpy(dst, src, data_size);
  return Error::Ok;
}

// =====================================================================
// MPS device-type override
// =====================================================================

__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_mps() {
  return 13; // Matches c10/core/DeviceType.h::MPS
}

AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type) {
  (void)tensor;
  if (ret_device_type == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_device_type = aoti_torch_device_type_mps();
  return Error::Ok;
}

} // extern "C"

// ---------------------------------------------------------------------
// Missing dtype shim (workaround for upstream gap)
//
// backends/aoti/common_shims_slim.cpp defines aoti_torch_dtype_float32(),
// _bfloat16(), _int*(), etc. but not _float16. Without this symbol the
// AOTI-generated .so for an fp16 model dlopens to a partially-resolved
// state and dies with SIGSEGV on first use of the missing trampoline.
//
// We define it locally so the symbol resolves at dlopen time. Note: actual
// fp16 execution is NOT supported because slim::c10::ScalarType doesn't
// include Half (creating a SlimTensor with dtype=5 will assert in
// check_supportive). This shim only satisfies the linker.
extern "C" {
int32_t aoti_torch_dtype_float16() {
  return 5;  // PyTorch's float16 dtype code (c10::ScalarType::Half)
}
} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
