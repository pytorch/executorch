/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/tensor_attribute.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/platform/log.h>
#include <algorithm>
#include <cstdint> // Ensure we have int64_t, int32_t definitions
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>

#include <vector>

namespace executorch {
namespace backends {
namespace metal {

// Import all from aoti namespace
using namespace executorch::backends::aoti;

// Global storage for tensors and their metadata.
// Maps raw Tensor* → shared_ptr<Tensor> for O(1) lookup/deletion.
std::unordered_map<Tensor*, std::shared_ptr<Tensor>> tensors;

// Reference counting for memory addresses
// Maps memory address to number of tensors using it
// Special value: NOT_OWN (-1) means tensor never owns the memory
constexpr int32_t NOT_OWN = -1;
std::unordered_map<void*, int32_t> memory_to_n_tensor;

// Every handle into memory the runtime owns holds a count on that allocation
// in memory_to_n_tensor. A handle whose own address is not an allocation (a
// view at an offset, or a handle made from one) finds only that address when
// it is deleted, so this maps it to the allocation its count went to.
std::unordered_map<Tensor*, void*> view_owner;

// Size of each CPU allocation the runtime owns, so that views of it can be
// bound into one Metal buffer over all of it (metal_register_cpu_view).
std::unordered_map<void*, size_t> cpu_allocation_bytes;

namespace {

// The owned allocation that `handle`, with data at `data_ptr`, lives in, or
// null for memory the runtime does not own, such as a model's constants.
void* owning_allocation(Tensor* handle, void* data_ptr) {
  auto owner = view_owner.find(handle);
  if (owner != view_owner.end()) {
    return owner->second;
  }
  auto memory = memory_to_n_tensor.find(data_ptr);
  if (memory != memory_to_n_tensor.end() && memory->second != NOT_OWN) {
    return data_ptr;
  }
  return nullptr;
}

// Takes a count on `owner`, if any, for a new handle with data at `data_ptr`.
void hold_allocation(Tensor* handle, void* data_ptr, void* owner) {
  if (owner == nullptr) {
    return;
  }
  memory_to_n_tensor[owner] += 1;
  if (data_ptr != owner) {
    view_owner[handle] = owner;
  }
}

// Wraps `data` in a tensor whose strides are the ones given. from_blob() does
// not keep the strides it is handed: it sorts them into a dim order and derives
// the strides again from that. A dimension of size 1 has the same stride as the
// dimension outside it, and the sort leaves such a tie in index order, so
// channels-last strides {63, 1, 9, 1} of a {2, 1, 7, 9} tensor come back as
// {63, 9, 9, 1}. The memory is the same, but the layout is no longer
// recognizable, and the convolution needs it to pick its output layout. Putting
// the size-1 dimension last among equal strides gives back the original ones.
std::shared_ptr<Tensor> make_strided_tensor(
    void* data,
    std::vector<aten::SizesType> sizes,
    std::vector<aten::StridesType> strides,
    aten::ScalarType scalar_type) {
  std::vector<aten::DimOrderType> dim_order(sizes.size());
  std::iota(dim_order.begin(), dim_order.end(), 0);
  std::stable_sort(dim_order.begin(), dim_order.end(), [&](size_t a, size_t b) {
    if (strides[a] != strides[b]) {
      return strides[a] > strides[b];
    }
    return sizes[a] != 1 && sizes[b] == 1;
  });
  return executorch::extension::for_blob(data, std::move(sizes), scalar_type)
      .dim_order(std::move(dim_order))
      .strides(std::move(strides))
      .make_tensor_ptr();
}

} // namespace

extern "C" {

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
  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: entered");

  (void)device_type;
  (void)opaque_metadata;
  (void)layout;
  (void)opaque_metadata_size;

  // Validate input parameters first
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: data pointer is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: sizes_ptr is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: ret_new_tensor is null");

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Handle storage offset by adjusting the data pointer
  void* adjusted_data = static_cast<char*>(data) +
      (storage_offset * dtype_to_element_size(dtype));

  ET_LOG(
      Debug,
      "aoti_torch_create_tensor_from_blob_v2: original_data=%p, storage_offset=%lld, element_size=%zu, adjusted_data=%p",
      data,
      storage_offset,
      dtype_to_element_size(dtype),
      adjusted_data);

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Log if the tensor is contiguous
  if (is_contiguous_tensor(sizes, strides)) {
    ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: contiguous tensor");
  } else {
    ET_LOG(
        Debug, "aoti_torch_create_tensor_from_blob_v2: non-contiguous tensor");
  }

  // ETensor creation
  // Note: We're NOT copying the data, just wrapping it
  auto tensor = make_strided_tensor(
      adjusted_data, sizes, strides, dtype_to_scalar_type(dtype));

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create tensor from blob");

  // Store the tensor so it doesn't get destroyed
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();

  // Check if this memory address is already being tracked
  auto memory_it = memory_to_n_tensor.find(adjusted_data);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it == memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is already being tracked by another tensor",
      adjusted_data);

  // Mark this memory as NOT_OWN since tensor created from blob never owns
  // memory
  memory_to_n_tensor[adjusted_data] = NOT_OWN;

  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: successful");
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
  ET_LOG(Debug, "aoti_torch_empty_strided: entered");

  // This requires us to reserve device memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

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
    if (!ptr) {
      ET_LOG(Error, "Failed to allocate %lld bytes on Metal device", nbytes);
      return Error::MemoryAllocationFailed;
    }
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match device requirements
    int result = posix_memalign(&ptr, 16, nbytes);
    ET_CHECK_OR_RETURN_ERROR(
        result == 0,
        MemoryAllocationFailed,
        "Failed to allocate aligned CPU memory");
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr,
        MemoryAllocationFailed,
        "Failed to call posix_memalign");
    ET_LOG(Debug, "Allocated %lld bytes on CPU", nbytes);
    cpu_allocation_bytes[ptr] = static_cast<size_t>(nbytes);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        false,
        NotImplemented,
        "Need to implement empty_strided for non-CUDA non-CPU device type %d",
        device_type);
  }

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Log if the tensor is contiguous
  if (is_contiguous_tensor(sizes, strides)) {
    ET_LOG(Debug, "aoti_torch_empty_strided: contiguous tensor");
  } else {
    ET_LOG(Debug, "aoti_torch_empty_strided: non-contiguous tensor");
  }

  // ETensor creation
  // Note: We're NOT copying the data, just wrapping it
  executorch::aten::ScalarType scalar_type = dtype_to_scalar_type(dtype);
  auto tensor = make_strided_tensor(ptr, sizes, strides, scalar_type);

  // Store the tensor so it doesn't get destroyed
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();

  // This tensor owns the memory it allocated, set reference count to 1
  memory_to_n_tensor[ptr] = 1;

  ET_LOG(Debug, "aoti_torch_empty_strided: successful");
  return Error::Ok;
}

// Drops one count on owned memory and frees it when none is left.
static AOTITorchError release_memory(void* data_ptr) {
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end() && memory_it->second > 0,
      Internal,
      "Internal error: releasing memory %p that is not owned",
      data_ptr);
  if (memory_it->second > 1) {
    memory_it->second -= 1;
    return Error::Ok;
  }
  auto cpu_allocation = cpu_allocation_bytes.find(data_ptr);
  if (cpu_allocation != cpu_allocation_bytes.end()) {
    // The GPU only reaches CPU memory through the buffer of a region of it
    // (metal_register_cpu_view). If there is one, work queued through it has to
    // finish before the memory goes; otherwise there is nothing to wait for.
    metal_release_cpu_region(data_ptr);
    cpu_allocation_bytes.erase(cpu_allocation);
    free(data_ptr);
    ET_LOG(Debug, "aoti_torch_delete_tensor_object: freeing CPU memory");
  } else if (metal_is_device_pointer(data_ptr)) {
    metal_deallocate_buffer(data_ptr);
  } else {
    free(data_ptr);
    ET_LOG(Debug, "aoti_torch_delete_tensor_object: freeing CPU memory");
  }
  memory_to_n_tensor.erase(memory_it);
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  ET_LOG(Debug, "aoti_torch_delete_tensor_object: entered");

  if (tensor == nullptr) {
    ET_LOG(Debug, "aoti_torch_delete_tensor_object: null tensor");
    return Error::Ok;
  }

  // O(1) lookup by raw pointer
  auto it = tensors.find(tensor);
  ET_CHECK_OR_RETURN_ERROR(
      it != tensors.end(), InvalidArgument, "Didn't find tensor %p", tensor);

  const auto& tensor_ptr = it->second;
  void* data_ptr = tensor_ptr->mutable_data_ptr();
  metal_forget_strided_view(tensor);

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      Internal,
      "Internal error: memory not found during deletion");

  if (memory_it->second == NOT_OWN) {
    // No-op unless this tensor is a view. With the last handle at a view's
    // address gone, nothing is there any more: a later allocation can land on
    // that address.
    if (metal_unregister_view(data_ptr)) {
      memory_to_n_tensor.erase(memory_it);
    }
    // Give back the count the view held on the allocation it lives in.
    auto owner = view_owner.find(tensor);
    if (owner != view_owner.end()) {
      void* allocation = owner->second;
      view_owner.erase(owner);
      ET_CHECK_OK_OR_RETURN_ERROR(release_memory(allocation));
    }
    tensors.erase(it);
    ET_LOG(
        Debug,
        "aoti_torch_delete_tensor_object: tensor doesn't own memory, skipping free");
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(release_memory(data_ptr));

  tensors.erase(it);
  ET_LOG(Debug, "aoti_torch_delete_tensor_object: successful");
  return Error::Ok;
}

AOTITorchError aoti_torch_copy_(
    AOTITensorHandle self,
    AOTITensorHandle src,
    int32_t non_blocking) {
  ET_LOG(Debug, "aoti_torch_copy_: entered");

  (void)non_blocking;

  // Check for null pointers first
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_copy_ failed: self tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      src != nullptr,
      InvalidArgument,
      "aoti_torch_copy_ failed: src tensor is null");

  // A strided view carries the strides of a packed tensor rather than its own,
  // so copying by those strides would write the wrong elements.
  ET_CHECK_OR_RETURN_ERROR(
      !metal_is_strided_view(self),
      NotSupported,
      "aoti_torch_copy_ does not support copying into a view that is not densely packed");

  // Get dtype information and validate compatibility
  int32_t self_dtype, src_dtype;
  aoti_torch_get_dtype(self, &self_dtype);
  aoti_torch_get_dtype(src, &src_dtype);

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(self_dtype));

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(src_dtype));

  // Check dtype compatibility - both tensors must have the same dtype
  ET_CHECK_OR_RETURN_ERROR(
      self_dtype == src_dtype,
      InvalidArgument,
      "dtype mismatch. self.dtype=%d, src.dtype=%d. aoti_torch_copy_ requires same dtypes",
      self_dtype,
      src_dtype);

  // Check total number of elements compatibility (PyTorch copy_ behavior)
  int64_t self_numel = self->numel();
  int64_t src_numel = src->numel();

  ET_CHECK_OR_RETURN_ERROR(
      self_numel == src_numel,
      InvalidArgument,
      "numel mismatch. self.numel()=%ld, src.numel()=%ld",
      self_numel,
      src_numel);

  // Get tensor metadata
  int64_t* self_strides;
  int64_t* src_strides;
  aoti_torch_get_strides(self, &self_strides);
  aoti_torch_get_strides(src, &src_strides);

  int64_t* self_sizes;
  int64_t* src_sizes;
  aoti_torch_get_sizes(self, &self_sizes);
  aoti_torch_get_sizes(src, &src_sizes);

  // Determine device locations
  bool srcIsDevice = false;
  bool dstIsDevice = false;

  // Check if pointers are Metal device pointers
  if (!srcIsDevice) {
    srcIsDevice = metal_is_device_pointer(const_cast<void*>(src->data_ptr()));
  }
  if (!dstIsDevice) {
    dstIsDevice = metal_is_device_pointer(self->mutable_data_ptr());
  }

  // Check if tensors have the same schema (sizes, strides, dtype) for fast path
  // TODO: This should be improved to catch cases like (4, 1, 5) -> (4, 5)
  bool same_schema = true;
  for (int i = 0; i < self->dim(); i++) {
    // A dimension of size 1 in both does not change where the elements are.
    if (self_sizes[i] == 1 && src_sizes[i] == 1) {
      continue;
    }
    if (self_strides[i] != src_strides[i]) {
      same_schema = false;
      break;
    }
  }

  size_t total_bytes = src->nbytes();
  int64_t total_elements = self->numel();

  if (same_schema && metal_is_strided_view(src)) {
    // E.g. a model output that is a chunk of a larger buffer. The strides
    // compared above are the packed ones `src` carries, so `self` is packed
    // the same way and takes the view's elements in order.
    ET_CHECK_OR_RETURN_ERROR(
        metal_copy_strided_view(*src, self->mutable_data_ptr()),
        Internal,
        "aoti_torch_copy_: failed to copy a view that is not densely packed");
  } else if (same_schema) {
    int result = metal_copy_memory(
        self->mutable_data_ptr(),
        src->data_ptr(),
        total_bytes,
        srcIsDevice,
        dstIsDevice);
    if (result != 0) {
      ET_LOG(Error, "metal_copy_memory failed with status %d", result);
      return Error::Internal;
    }
  } else {
    ET_LOG(Error, "Layout conversion not supported");
    return Error::NotImplemented;
  }

  ET_LOG(Debug, "aoti_torch_copy_: successful");
  return Error::Ok;
}

// Check if a strided view is densely packed (no holes in memory).
// A densely packed tensor's storage extent equals its numel.
static bool is_packed_strides(
    const std::vector<aten::SizesType>& sizes,
    const std::vector<aten::StridesType>& strides) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  if (ndim == 0)
    return true;

  // Compute numel
  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    numel *= sizes[i];
  }
  if (numel <= 1)
    return true;

  // Compute storage extent: max offset + 1
  int64_t max_offset = 0;
  for (int64_t i = 0; i < ndim; i++) {
    if (sizes[i] > 1) {
      max_offset += static_cast<int64_t>(sizes[i] - 1) * strides[i];
    }
  }
  return (max_offset + 1) == numel;
}

// Materialize a non-packed strided view of CPU memory into a new contiguous
// Metal buffer. Copies elements from source using strided access. The caller
// must free the returned buffer. On failure returns nullptr.
static void* materialize_packed(
    void* src,
    const std::vector<aten::SizesType>& sizes,
    const std::vector<aten::StridesType>& strides,
    size_t element_size) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    numel *= sizes[i];
  }

  bool dst_may_be_in_use = false;
  void* dst = metal_allocate_buffer_tracking_use(
      numel * element_size, &dst_may_be_in_use);
  if (!dst)
    return nullptr;

  // The copy is made on the CPU, so queued GPU work on either side has to be
  // done first: writes to the source, if it lies in memory the GPU can write
  // (other CPU memory is only ever bound by copy, see
  // ETMetalKernelFunction::setArg), and uses of `dst`, if its buffer was
  // recycled before the stream last waited.
  int64_t extent = 1;
  for (int64_t i = 0; i < ndim; i++) {
    if (sizes[i] > 1) {
      extent += static_cast<int64_t>(sizes[i] - 1) * strides[i];
    }
  }
  auto* stream = getCurrentMetalStream();
  if (stream &&
      (dst_may_be_in_use ||
       metal_overlaps_gpu_memory(src, extent * element_size))) {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  }

  // Element-by-element strided copy
  char* src_bytes = static_cast<char*>(src);
  char* dst_bytes = static_cast<char*>(dst);
  std::vector<int64_t> coord(ndim, 0);
  for (int64_t flat = 0; flat < numel; flat++) {
    // Compute source offset from strides
    int64_t src_offset = 0;
    for (int64_t d = 0; d < ndim; d++) {
      src_offset += coord[d] * strides[d];
    }
    std::memcpy(
        dst_bytes + flat * element_size,
        src_bytes + src_offset * element_size,
        element_size);

    // Increment coordinate (last dim fastest)
    for (int64_t d = ndim - 1; d >= 0; d--) {
      if (++coord[d] < sizes[d])
        break;
      coord[d] = 0;
    }
  }
  return dst;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {
  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: entered");

  // Validate input parameters first
  ET_CHECK_OR_RETURN_ERROR(
      ndim >= 0,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: ndim must be >= 0, got %lld",
      ndim);

  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: self tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: sizes_ptr is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: ret_new_tensor is null");

  // Get the device info from the source tensor to perform device_index
  // validation
  int32_t device_type = 0;
  int32_t device_index = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_type(self, &device_type));

  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_index(self, &device_index));

  // Ensure device_index is always 0
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // Get the dtype from the source tensor
  int32_t dtype = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(self, &dtype));

  // Validate dtype using SupportedDTypes
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Get the original data pointer from the source tensor
  void* data_ptr = self->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  // Check if the given memory is in the map, if not return error
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  // Handle storage offset by adjusting the data pointer
  size_t element_size = dtype_to_element_size(dtype);
  void* adjusted_data =
      static_cast<char*>(data_ptr) + (storage_offset * element_size);

  // Convert sizes using utility function from utils.h
  std::vector<aten::SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // An empty view reads nothing. Its offset can put it at the very end of its
  // buffer, which may be where the next allocation starts; point it at the
  // start of its parent instead, so that it is not taken for a view there.
  int64_t view_numel = 1;
  for (auto size : sizes) {
    view_numel *= size;
  }
  if (view_numel == 0) {
    adjusted_data = data_ptr;
  }

  // Convert strides using utility function from utils.h
  std::vector<aten::StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // A view that is not densely packed (e.g. one half of a channels-last tensor
  // chunked along C) cannot be described by a tensor's strides here, so the
  // tensor gets the strides of a packed one.
  void* tensor_data = adjusted_data;
  bool owns_buffer = false;
  bool strided_view = false;
  std::vector<int64_t> view_sizes;
  std::vector<int64_t> view_strides;
  if (!is_packed_strides(sizes, strides)) {
    if (metal_is_device_pointer(data_ptr) && !metal_is_cpu_memory(data_ptr)) {
      // Leave the view inside its parent. Kernels generated by inductor index
      // it with the strides they were compiled with and write through it (a cat
      // is filled through such views), so a packed copy would both lose those
      // writes and be indexed past its end. Ops that need dense input take a
      // packed copy when they run (metal_packed_copy_of_strided_view).
      ET_LOG(
          Debug,
          "aoti_torch__reinterpret_tensor: non-packed strides, keeping the view in its parent's buffer");
      strided_view = true;
      view_sizes.assign(sizes.begin(), sizes.end());
      view_strides.assign(strides.begin(), strides.end());
    } else {
      ET_LOG(
          Debug,
          "aoti_torch__reinterpret_tensor: non-packed strides, "
          "materializing to packed buffer");
      tensor_data =
          materialize_packed(adjusted_data, sizes, strides, element_size);
      ET_CHECK_OR_RETURN_ERROR(
          tensor_data != nullptr,
          MemoryAllocationFailed,
          "Failed to materialize non-packed tensor");
      owns_buffer = true;
    }

    // Compute contiguous strides for the packed layout
    strides.resize(ndim);
    if (ndim > 0) {
      strides[ndim - 1] = 1;
      for (int64_t i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * sizes[i + 1];
      }
    }
  }

  std::shared_ptr<Tensor> tensor = make_strided_tensor(
      tensor_data, sizes, strides, dtype_to_scalar_type(dtype));

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "Failed to create reinterpreted tensor view");

  if (owns_buffer) {
    // The materialized buffer is a new allocation owned by this tensor
    memory_to_n_tensor[tensor_data] = 1;
  } else {
    if (adjusted_data != data_ptr) {
      ET_LOG(
          Debug,
          "aoti_torch__reinterpret_tensor: Adjusted original_data=%p, "
          "storage_offset=%lld, element_size=%zu, adjusted_data=%p",
          data_ptr,
          storage_offset,
          element_size,
          adjusted_data);

      if (metal_is_device_pointer(data_ptr) && !metal_is_cpu_memory(data_ptr)) {
        // The view shares its parent's Metal buffer and is bound at an
        // offset. It must not get an MTLBuffer of its own: Metal would treat
        // the two as unrelated, and inductor both reads views of a buffer
        // another op is still writing and fills a buffer (e.g. the result of
        // a cat) by writing through views of it.
        ET_CHECK_OR_RETURN_ERROR(
            metal_register_view(adjusted_data, data_ptr),
            Internal,
            "Failed to register adjusted_data=%p as a view of %p",
            adjusted_data,
            data_ptr);
      } else {
        // CPU memory has no Metal buffer of its own. Views of it are bound
        // into a no-copy buffer over the CPU region they belong to, shared by
        // all of them, so that Metal orders their uses: for memory the runtime
        // allocated, the whole allocation; otherwise the region of `self`.
        void* region = owning_allocation(self, data_ptr);
        size_t region_nbytes = 0;
        auto allocation = region != nullptr ? cpu_allocation_bytes.find(region)
                                            : cpu_allocation_bytes.end();
        const bool owned = allocation != cpu_allocation_bytes.end();
        if (owned) {
          region_nbytes = allocation->second;
        } else if (!metal_cpu_view_region(data_ptr, &region)) {
          region = data_ptr;
          region_nbytes = self->nbytes();
        }
        ET_CHECK_OR_RETURN_ERROR(
            metal_register_cpu_view(
                adjusted_data, tensor->nbytes(), region, region_nbytes, owned),
            Internal,
            "Failed to wrap the CPU memory of adjusted_data=%p in a Metal buffer",
            adjusted_data);
      }

      memory_to_n_tensor[adjusted_data] = NOT_OWN;
    } else {
      // Another handle at the address of `self`. If `self` is a view, deleting
      // either handle must leave the view registered for the other one.
      metal_retain_view(data_ptr);
    }

    // The new handle keeps the allocation it lives in alive, including when
    // `self` is itself a view.
    hold_allocation(
        tensor.get(), adjusted_data, owning_allocation(self, data_ptr));
  }

  // Only now that nothing can fail: a handle whose registration failed must
  // not be left behind for cleanup to unregister.
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  if (strided_view) {
    metal_record_strided_view(
        tensor.get(), std::move(view_sizes), std::move(view_strides));
  }

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: successful");
  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  ET_LOG(Debug, "aoti_torch_new_tensor_handle: entered");

  // Validate input parameters
  ET_CHECK_OR_RETURN_ERROR(
      orig_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle failed: orig_handle is null");

  ET_CHECK_OR_RETURN_ERROR(
      new_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle failed: new_handle is null");

  // Get metadata from the original tensor
  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  int32_t dtype;
  int32_t device_type;
  int32_t device_index;

  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_sizes(orig_handle, &sizes_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_strides(orig_handle, &strides_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(orig_handle, &dtype));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_device_type(orig_handle, &device_type));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_device_index(orig_handle, &device_index));

  int64_t ndim = orig_handle->dim();

  // Validate dtype
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Ensure device_index is always 0
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // Get the original data pointer from the source tensor
  void* data_ptr = orig_handle->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  // Check if the given memory is in the map
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  // Convert sizes and strides to vectors
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor that shares the same memory as the original
  // This is similar to PyTorch's Tensor copy constructor - creates a new
  // tensor object that shares the same underlying storage
  std::shared_ptr<Tensor> tensor = make_strided_tensor(
      data_ptr, // Share the same memory from source tensor
      sizes, // Same sizes as original
      strides, // Same strides as original
      dtype_to_scalar_type(dtype) // Same dtype as original
  );

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create new tensor handle");

  // Store the tensor so it doesn't get destroyed
  tensors[tensor.get()] = tensor;

  *new_handle = tensor.get();

  metal_share_strided_view(orig_handle, tensor.get());

  // If the original is a view into a Metal buffer, the new handle is one too,
  // and deleting either must leave the view registered for the other one.
  metal_retain_view(data_ptr);

  // The new handle keeps the allocation the original lives in alive.
  hold_allocation(
      tensor.get(), data_ptr, owning_allocation(orig_handle, data_ptr));

  ET_LOG(Debug, "aoti_torch_new_tensor_handle: successful");
  return Error::Ok;
}

// Cleanup function for clearing global state
void cleanup_memory() {
  // Use aoti_torch_delete_tensor_object to properly delete each tensor.
  // Collect keys first since deletion modifies the map.
  std::vector<Tensor*> tensor_ptrs;
  tensor_ptrs.reserve(tensors.size());
  for (const auto& entry : tensors) {
    tensor_ptrs.push_back(entry.first);
  }

  for (Tensor* tensor_ptr : tensor_ptrs) {
    aoti_torch_delete_tensor_object(tensor_ptr);
  }

  // tensors map should now be empty, but ensure it's cleared
  tensors.clear();

  // Tensors created from a blob are tracked as NOT_OWN and
  // aoti_torch_delete_tensor_object leaves their address in the map, since
  // several of them may alias it. With every tensor gone nothing is tracked
  // anymore, and a stale entry would make the next model fail to load as soon
  // as its constants land on an address used before.
  memory_to_n_tensor.clear();
  view_owner.clear();
  cpu_allocation_bytes.clear();

  // Clean up Metal resources
  metal_cleanup_resources();

  ET_LOG(Info, "Cleared all tensors and Metal resources");
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
