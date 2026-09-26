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
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
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
#include <unordered_set>

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

// Handles that took a registration on the view at their address
// (metal_register_view, metal_register_cpu_view or metal_retain_view). Several
// handles can share an address, e.g. a view and a blob, so only these give one
// back when they are deleted.
std::unordered_set<Tensor*> view_handles;

// How many handles are at each address the runtime does not own (blobs and
// views): the address stays tracked while any of them lives.
std::unordered_map<void*, int32_t> not_own_handles;

// Size of each CPU allocation the runtime owns, so that views of it can be
// bound into one Metal buffer over all of it (metal_register_cpu_view).
std::unordered_map<void*, size_t> cpu_allocation_bytes;

namespace {

// The first and one past the last byte a view with these sizes and strides
// touches, relative to its start, in `*lowest` and `*end`. Returns false if
// they do not fit in 64 bits.
bool view_byte_span(
    const std::vector<aten::SizesType>& sizes,
    const std::vector<aten::StridesType>& strides,
    size_t element_size,
    int64_t* lowest,
    int64_t* end) {
  int64_t low = 0;
  int64_t high = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    const int64_t span = static_cast<int64_t>(sizes[i] - 1) * strides[i];
    if (__builtin_add_overflow(
            span < 0 ? low : high, span, span < 0 ? &low : &high)) {
      return false;
    }
  }
  const auto size = static_cast<int64_t>(element_size);
  return !__builtin_mul_overflow(low, size, lowest) &&
      !__builtin_add_overflow(high, int64_t{1}, &high) &&
      !__builtin_mul_overflow(high, size, end);
}

// Whether sizes and strides as AOTInductor passes them (int64) fit the int32
// ones a tensor holds here, sizes being non-negative. `strides` may be null.
bool fits_tensor(int64_t ndim, const int64_t* sizes, const int64_t* strides) {
  // A dim of size 1 never steps, so its stride does not have to fit.
  for (int64_t i = 0; i < ndim; i++) {
    if (sizes[i] < 0 || sizes[i] > INT32_MAX ||
        (strides != nullptr && sizes[i] > 1 &&
         (strides[i] < INT32_MIN || strides[i] > INT32_MAX))) {
      return false;
    }
  }
  // Without strides, the contiguous ones are computed and must fit too.
  int64_t stride = 1;
  for (int64_t i = ndim - 1; strides == nullptr && i >= 0; i--) {
    if (sizes[i] > 1 && stride > INT32_MAX) {
      return false;
    }
    if (sizes[i] != 0 && __builtin_mul_overflow(stride, sizes[i], &stride)) {
      stride = INT64_MAX;
    }
  }
  return true;
}

// A tensor's strides: the ones given, or the contiguous ones. The stride of a
// dim of size 1 that does not fit int32 is never used, and is 1 instead, which
// keeps that dim innermost; a truncated one could make it look outermost.
// Call after fits_tensor.
std::vector<aten::StridesType>
tensor_strides(int64_t ndim, const int64_t* sizes, const int64_t* strides) {
  std::vector<aten::StridesType> result(ndim);
  int64_t contiguous = 1;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    const int64_t stride = strides != nullptr ? strides[i] : contiguous;
    result[i] = stride < INT32_MIN || stride > INT32_MAX
        ? 1
        : static_cast<aten::StridesType>(stride);
    if (sizes[i] != 0 &&
        __builtin_mul_overflow(contiguous, sizes[i], &contiguous)) {
      contiguous = INT64_MAX;
    }
  }
  return result;
}

// `data` moved by `storage_offset` elements, or false if that overflows.
bool offset_pointer(
    void* data,
    int64_t storage_offset,
    size_t element_size,
    void** adjusted) {
  int64_t offset_bytes = 0;
  uintptr_t address = 0;
  if (__builtin_mul_overflow(
          storage_offset, static_cast<int64_t>(element_size), &offset_bytes) ||
      __builtin_add_overflow(
          reinterpret_cast<uintptr_t>(data), offset_bytes, &address)) {
    return false;
  }
  *adjusted = reinterpret_cast<void*>(address);
  return true;
}

// The number of elements of a tensor with these sizes, or false if it does
// not fit in 64 bits.
bool checked_numel(const int64_t* sizes, int64_t ndim, int64_t* numel) {
  *numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    if (__builtin_mul_overflow(*numel, sizes[i], numel)) {
      return false;
    }
  }
  return true;
}

// Whether a region over memory the runtime did not allocate, starting at
// `region` and covering at least `region_nbytes` and the `view_nbytes` at
// `view`, may take in what it would newly cover. Such a region grows with its
// views, so it must not take in a CPU allocation of the runtime's, or a
// tensor that lies there without being registered in it: that tensor would be
// bound by copy, unordered with work through the region.
bool region_can_cover(
    void* region,
    size_t region_nbytes,
    void* view,
    size_t view_nbytes) {
  void* base = nullptr;
  bool cpu = false;
  size_t existing = 0;
  if (!metal_find_memory(region, &base, &cpu, &existing) || base != region) {
    existing = 0;
  }
  const auto* begin = static_cast<const uint8_t*>(region);
  const auto* end = std::max(
      begin + std::max(existing, region_nbytes),
      static_cast<const uint8_t*>(view) + view_nbytes);
  if (end <= begin + existing) {
    return true;
  }
  for (const auto& allocation : cpu_allocation_bytes) {
    const auto* start = static_cast<const uint8_t*>(allocation.first);
    if (start < end && begin < start + allocation.second) {
      return false;
    }
  }
  for (const auto& entry : tensors) {
    void* data = entry.first->mutable_data_ptr();
    const auto* start = static_cast<const uint8_t*>(data);
    void* data_region = nullptr;
    if (start < end && begin < start + entry.first->nbytes() &&
        data != region &&
        !(metal_cpu_view_region(data, &data_region) && data_region == region)) {
      return false;
    }
  }
  return true;
}

// Records one more handle at `ptr`, memory the runtime does not own.
void add_not_own_handle(void* ptr) {
  memory_to_n_tensor[ptr] = NOT_OWN;
  not_own_handles[ptr] += 1;
}

// Whether any of the `nbytes` at `ptr` lie in a CPU allocation of the
// runtime's.
bool overlaps_cpu_allocation(const void* ptr, size_t nbytes) {
  const auto* begin = static_cast<const uint8_t*>(ptr);
  for (const auto& allocation : cpu_allocation_bytes) {
    const auto* start = static_cast<const uint8_t*>(allocation.first);
    if (start < begin + nbytes && begin < start + allocation.second) {
      return true;
    }
  }
  return false;
}

// The runtime's own CPU allocation that `ptr` lies in, or null.
void* cpu_allocation_containing(const void* ptr, size_t* nbytes) {
  const auto* p = static_cast<const uint8_t*>(ptr);
  for (const auto& allocation : cpu_allocation_bytes) {
    const auto* begin = static_cast<const uint8_t*>(allocation.first);
    if (begin <= p && p < begin + allocation.second) {
      *nbytes = allocation.second;
      return allocation.first;
    }
  }
  return nullptr;
}

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

// The dim order of a tensor with these strides, in `*dim_order`. from_blob()
// does not keep the strides it is handed: it sorts them into a dim order and
// derives the strides again from that. A dimension of size 1 has the same
// stride as the dimension outside it, and the sort leaves such a tie in index
// order, so channels-last strides {63, 1, 9, 1} of a {2, 1, 7, 9} tensor come
// back as {63, 9, 9, 1}. The memory is the same, but the layout is no longer
// recognizable, and the convolution needs it to pick its output layout. Putting
// the size-1 dimension last among equal strides gives back the original ones.
// Returns false for negative sizes, and for strides that are not those of a
// dense tensor in that order, such as a view with holes, overlapping dims or
// a stride of 0:
// make_tensor_ptr aborts on them. A tensor with no elements has no layout, so
// any strides do; `*dense` gets the strides make_tensor_ptr accepts for it.
bool dense_dim_order(
    const std::vector<aten::SizesType>& sizes,
    const std::vector<aten::StridesType>& strides,
    std::vector<aten::DimOrderType>* dim_order,
    std::vector<aten::StridesType>* dense) {
  if (std::any_of(
          sizes.begin(), sizes.end(), [](auto size) { return size < 0; })) {
    return false;
  }
  dim_order->resize(sizes.size());
  std::iota(dim_order->begin(), dim_order->end(), 0);
  std::stable_sort(
      dim_order->begin(), dim_order->end(), [&](size_t a, size_t b) {
        if (strides[a] != strides[b]) {
          return strides[a] > strides[b];
        }
        return sizes[a] != 1 && sizes[b] == 1;
      });
  // dim_order_to_stride multiplies the int32 sizes of all but the outermost
  // dim: their product, the outermost stride, must fit.
  int32_t product = 1;
  for (size_t i = 1; i < dim_order->size(); i++) {
    const auto size = sizes[(*dim_order)[i]];
    if (size != 0 && __builtin_mul_overflow(product, size, &product)) {
      return false;
    }
  }
  dense->resize(sizes.size());
  if (executorch::runtime::dim_order_to_stride(
          sizes.data(), dim_order->data(), sizes.size(), dense->data()) !=
      executorch::runtime::Error::Ok) {
    return false;
  }
  if (std::find(sizes.begin(), sizes.end(), 0) != sizes.end()) {
    return true;
  }
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] != 1 && strides[i] != (*dense)[i]) {
      return false;
    }
  }
  return true;
}

// Wraps `data` in a tensor whose strides are the ones given, keeping them
// (see dense_dim_order), apart from those of size-1 dims, which make_tensor_ptr
// sets itself. Returns null for strides a tensor cannot have.
std::shared_ptr<Tensor> make_strided_tensor(
    void* data,
    std::vector<aten::SizesType> sizes,
    std::vector<aten::StridesType> strides,
    aten::ScalarType scalar_type) {
  std::vector<aten::DimOrderType> dim_order;
  std::vector<aten::StridesType> dense;
  if (!dense_dim_order(sizes, strides, &dim_order, &dense)) {
    ET_LOG(Error, "The strides of a tensor must be those of a dense one");
    return nullptr;
  }
  if (std::find(sizes.begin(), sizes.end(), 0) != sizes.end()) {
    strides = std::move(dense);
  }
  return executorch::extension::for_blob(data, std::move(sizes), scalar_type)
      .dim_order(std::move(dim_order))
      .strides(std::move(strides))
      .make_tensor_ptr();
}

// An empty tensor with no memory, e.g. AOTInductor's handle for a constant
// of size 0, whose data pointer is null. It points nowhere and is tracked in
// nothing but `tensors`.
AOTITorchError make_memoryless_tensor(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    AOTITensorHandle* ret_new_tensor) {
  int64_t numel = 0;
  ET_CHECK_OR_RETURN_ERROR(
      ndim >= 0 && !(sizes_ptr == nullptr && ndim > 0) &&
          fits_tensor(ndim, sizes_ptr, strides_ptr) &&
          checked_numel(sizes_ptr, ndim, &numel) && numel == 0,
      InvalidArgument,
      "Only a tensor with no elements can have a null data pointer");
  auto tensor = make_strided_tensor(
      nullptr,
      convert_sizes_to_vector(ndim, sizes_ptr),
      tensor_strides(ndim, sizes_ptr, strides_ptr),
      dtype_to_scalar_type(dtype));
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create empty tensor");
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  return Error::Ok;
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
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: sizes_ptr is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: ret_new_tensor is null");

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));
  if (data == nullptr) {
    return make_memoryless_tensor(
        ndim, sizes_ptr, strides_ptr, dtype, ret_new_tensor);
  }
  ET_CHECK_OR_RETURN_ERROR(
      ndim >= 0 && fits_tensor(ndim, sizes_ptr, strides_ptr),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: sizes and strides must fit a "
      "tensor");

  // Handle storage offset by adjusting the data pointer
  void* adjusted_data = nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      offset_pointer(
          data, storage_offset, dtype_to_element_size(dtype), &adjusted_data),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: storage_offset overflows");

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
  auto strides = tensor_strides(ndim, sizes_ptr, strides_ptr);

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

  // Check if this memory address is already being tracked
  auto memory_it = memory_to_n_tensor.find(adjusted_data);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it == memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is already being tracked by another tensor",
      adjusted_data);

  // A blob can lie inside memory the runtime already binds for the GPU: a
  // Metal buffer, a CPU allocation of its own, or a CPU region. It is then
  // registered as a view of that memory, as aoti_torch__reinterpret_tensor
  // registers views, and holds the allocation it lies in. Otherwise kernels
  // would bind it by copy, graphs would not bind it at all, and a view of it
  // would get a Metal buffer of its own over bytes another buffer covers. This
  // is decided when the blob is made: a region made later over a blob of plain
  // CPU memory does not take it in, and views that would make two regions
  // overlap are refused (metal_register_cpu_view). AOTInductor does not make
  // blobs over CPU memory.
  // The tensor is dense (make_tensor_ptr checks its strides), so its bytes
  // are all it covers.
  int64_t numel = 0;
  int64_t checked_nbytes = 0;
  ET_CHECK_OR_RETURN_ERROR(
      checked_numel(sizes_ptr, ndim, &numel) &&
          !__builtin_mul_overflow(
              numel,
              static_cast<int64_t>(dtype_to_element_size(dtype)),
              &checked_nbytes),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: the size in bytes overflows");
  const auto nbytes = static_cast<size_t>(checked_nbytes);
  const auto* blob_end = static_cast<const uint8_t*>(adjusted_data) + nbytes;
  bool registered = false;
  void* owner = nullptr;
  void* key = nullptr;
  bool key_cpu = false;
  size_t key_nbytes = 0;
  if (metal_is_device_pointer(adjusted_data)) {
    // At the start of a buffer, e.g. a constant's: it must fit. At the start
    // of a CPU region, which no tensor tracks any more, it joins the region.
    const bool found =
        metal_find_memory(adjusted_data, &key, &key_cpu, &key_nbytes) &&
        key == adjusted_data;
    if (found && key_cpu) {
      ET_CHECK_OR_RETURN_ERROR(
          region_can_cover(key, 0, adjusted_data, nbytes) &&
              metal_register_cpu_view(
                  adjusted_data, nbytes, key, 0, /*owned=*/false),
          InvalidArgument,
          "Failed to register blob %p in the CPU region there",
          adjusted_data);
      registered = true;
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          !found || nbytes <= metal_constant_extent(key, key_nbytes),
          InvalidArgument,
          "Blob of %zu bytes at %p does not fit the Metal buffer there",
          nbytes,
          adjusted_data);
    }
  } else {
    size_t allocation_nbytes = 0;
    void* allocation =
        cpu_allocation_containing(adjusted_data, &allocation_nbytes);
    void* base = nullptr;
    bool cpu = false;
    size_t base_nbytes = 0;
    if (allocation != nullptr) {
      ET_CHECK_OR_RETURN_ERROR(
          blob_end <=
                  static_cast<const uint8_t*>(allocation) + allocation_nbytes &&
              metal_register_cpu_view(
                  adjusted_data,
                  nbytes,
                  allocation,
                  allocation_nbytes,
                  /*owned=*/true),
          InvalidArgument,
          "Blob of %zu bytes at %p does not fit the CPU allocation at %p",
          nbytes,
          adjusted_data,
          allocation);
      owner = allocation;
      registered = true;
    } else if (metal_find_memory(adjusted_data, &base, &cpu, &base_nbytes)) {
      if (cpu) {
        ET_CHECK_OR_RETURN_ERROR(
            region_can_cover(base, 0, adjusted_data, nbytes) &&
                metal_register_cpu_view(
                    adjusted_data, nbytes, base, 0, /*owned=*/false),
            InvalidArgument,
            "Failed to register blob %p in the CPU region at %p",
            adjusted_data,
            base);
      } else {
        ET_CHECK_OR_RETURN_ERROR(
            blob_end <= static_cast<const uint8_t*>(base) +
                        metal_constant_extent(base, base_nbytes) &&
                metal_register_view(adjusted_data, base),
            InvalidArgument,
            "Blob of %zu bytes at %p does not fit the Metal buffer at %p",
            nbytes,
            adjusted_data,
            base);
        // Only constants lie in a buffer inside another one, and neither of
        // those is owned: an owned `base` is the whole allocation.
        auto base_memory = memory_to_n_tensor.find(base);
        if (base_memory != memory_to_n_tensor.end() &&
            base_memory->second != NOT_OWN) {
          owner = base;
        }
      }
      registered = true;
    } else {
      // Not inside memory the runtime binds, so it must not reach into any:
      // it would be bound by copy, unordered with work on that memory.
      ET_CHECK_OR_RETURN_ERROR(
          nbytes == 0 ||
              (!overlaps_cpu_allocation(adjusted_data, nbytes) &&
               !metal_overlaps_gpu_memory(adjusted_data, nbytes)),
          InvalidArgument,
          "Blob of %zu bytes at %p reaches into memory the runtime binds",
          nbytes,
          adjusted_data);
    }
  }
  hold_allocation(tensor.get(), adjusted_data, owner);

  // Store the tensor so it doesn't get destroyed
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  if (registered) {
    view_handles.insert(tensor.get());
  }

  // Mark this memory as NOT_OWN since tensor created from blob never owns
  // memory
  add_not_own_handle(adjusted_data);

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

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr && ndim >= 0 &&
          !(sizes_ptr == nullptr && ndim > 0) &&
          fits_tensor(ndim, sizes_ptr, strides_ptr),
      InvalidArgument,
      "aoti_torch_empty_strided: invalid arguments, or sizes and strides that "
      "do not fit a tensor");

  // This requires us to reserve device memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  ET_CHECK_OR_RETURN_ERROR(
      checked_numel(sizes_ptr, ndim, &numel),
      InvalidArgument,
      "aoti_torch_empty_strided: the number of elements overflows");

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  size_t element_size = dtype_to_element_size(dtype);
  ET_CHECK_OR_RETURN_ERROR(
      element_size != 0,
      InvalidArgument,
      "Invalid element size for dtype: %d",
      dtype);
  // An empty tensor still gets memory: neither allocator hands out 0 bytes.
  int64_t nbytes = 0;
  ET_CHECK_OR_RETURN_ERROR(
      !__builtin_mul_overflow(
          numel, static_cast<int64_t>(element_size), &nbytes),
      InvalidArgument,
      "aoti_torch_empty_strided: the size in bytes overflows");
  nbytes = std::max<int64_t>(nbytes, 1);

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = tensor_strides(ndim, sizes_ptr, strides_ptr);

  // Checked before anything is allocated: a tensor cannot have other strides.
  std::vector<aten::DimOrderType> dim_order;
  std::vector<aten::StridesType> dense;
  ET_CHECK_OR_RETURN_ERROR(
      dense_dim_order(sizes, strides, &dim_order, &dense),
      InvalidArgument,
      "aoti_torch_empty_strided: strides are not those of a dense tensor");

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
  // An op can make its output a blob and then take ownership of it.
  not_own_handles.erase(data_ptr);
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
  if (data_ptr == nullptr) {
    // A tensor with no memory (make_memoryless_tensor).
    tensors.erase(it);
    return Error::Ok;
  }

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      Internal,
      "Internal error: memory not found during deletion");

  if (memory_it->second == NOT_OWN) {
    // Give back the count the handle held on the allocation it lives in. That
    // comes first: it is the only step that can fail.
    auto owner = view_owner.find(tensor);
    if (owner != view_owner.end()) {
      ET_CHECK_OK_OR_RETURN_ERROR(release_memory(owner->second));
      view_owner.erase(owner);
    }
    if (view_handles.erase(tensor) != 0) {
      metal_unregister_view(data_ptr);
    }
    // With the last handle at the address gone, nothing is there any more: a
    // later allocation can land on that address.
    auto handles = not_own_handles.find(data_ptr);
    if (handles == not_own_handles.end() || --handles->second <= 0) {
      if (handles != not_own_handles.end()) {
        not_own_handles.erase(handles);
      }
      memory_to_n_tensor.erase(data_ptr);
    }
    metal_forget_strided_view(tensor);
    tensors.erase(it);
    ET_LOG(
        Debug,
        "aoti_torch_delete_tensor_object: tensor doesn't own memory, skipping free");
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(release_memory(data_ptr));

  metal_forget_strided_view(tensor);
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
  // With different ranks the dims do not line up; the same bytes are only
  // meant if both are dense in row-major order.
  bool same_schema = self->dim() == src->dim() ||
      (is_row_major_dense(*self) && is_row_major_dense(*src));
  for (int i = 0; same_schema && self->dim() == src->dim() && i < self->dim();
       i++) {
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

// Check if a strided view is densely packed: no holes in memory, and no two
// elements at one address.
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

  // Dense in some dim order: no holes, and no two elements at one address.
  std::vector<aten::DimOrderType> dim_order;
  std::vector<aten::StridesType> dense;
  return dense_dim_order(sizes, strides, &dim_order, &dense);
}

// Materialize a non-packed strided view of CPU memory into a new contiguous
// Metal buffer. Copies elements from source using strided access.
// `in_metal_buffer` says the source is known to lie in a buffer the GPU uses,
// which spares looking it up. The caller must free the returned buffer. On
// failure returns nullptr.
static void* materialize_packed(
    bool in_metal_buffer,
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
  int64_t dst_nbytes = 0;
  if (__builtin_mul_overflow(
          numel, static_cast<int64_t>(element_size), &dst_nbytes)) {
    return nullptr;
  }
  void* dst =
      metal_allocate_buffer_tracking_use(dst_nbytes, &dst_may_be_in_use);
  if (!dst)
    return nullptr;

  // The copy is made on the CPU, so queued GPU work on either side has to be
  // done first (on the current stream: see getCurrentMetalStream): writes to
  // the source, if it lies in memory the GPU can write
  // (other CPU memory is only ever bound by copy, see
  // ETMetalKernelFunction::setArg), and uses of `dst`, if its buffer was
  // recycled before the stream last waited.
  int64_t lowest = 0;
  int64_t end = 0;
  if (!view_byte_span(sizes, strides, element_size, &lowest, &end)) {
    metal_deallocate_buffer(dst);
    return nullptr;
  }
  auto* stream = getCurrentMetalStream();
  if (stream &&
      (dst_may_be_in_use || in_metal_buffer ||
       metal_overlaps_gpu_memory(
           static_cast<char*>(src) + lowest,
           static_cast<size_t>(end - lowest)))) {
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
  if (data_ptr == nullptr) {
    // A view of a tensor with no memory has no elements either.
    return make_memoryless_tensor(
        ndim, sizes_ptr, strides_ptr, dtype, ret_new_tensor);
  }

  // Check if the given memory is in the map, if not return error
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  ET_CHECK_OR_RETURN_ERROR(
      ndim >= 0 && !(sizes_ptr == nullptr && ndim > 0) &&
          fits_tensor(ndim, sizes_ptr, strides_ptr),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: sizes and strides must fit a tensor");

  // Handle storage offset by adjusting the data pointer
  size_t element_size = dtype_to_element_size(dtype);
  void* adjusted_data = nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      offset_pointer(data_ptr, storage_offset, element_size, &adjusted_data),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: storage_offset overflows");

  // Convert sizes using utility function from utils.h
  std::vector<aten::SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  ET_CHECK_OR_RETURN_ERROR(
      std::all_of(
          sizes.begin(), sizes.end(), [](auto size) { return size >= 0; }),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: sizes must not be negative");

  // An empty view reads nothing. Its offset can put it at the very end of its
  // buffer, which may be where the next allocation starts; point it at the
  // start of its parent instead, so that it is not taken for a view there.
  int64_t view_numel = 1;
  for (auto size : sizes) {
    ET_CHECK_OR_RETURN_ERROR(
        !__builtin_mul_overflow(view_numel, int64_t{size}, &view_numel),
        InvalidArgument,
        "aoti_torch__reinterpret_tensor: the number of elements overflows");
  }
  if (view_numel == 0) {
    adjusted_data = data_ptr;
  }

  // Convert strides using utility function from utils.h
  std::vector<aten::StridesType> strides =
      tensor_strides(ndim, sizes_ptr, strides_ptr);

  // A view lies inside the memory `self` lies in, as far as it is known: a
  // CPU allocation of the runtime's or a Metal buffer. Past it is other
  // memory, e.g. the next allocation, which the view would be registered in.
  if (view_numel > 0) {
    int64_t lowest = 0;
    int64_t end = 0;
    ET_CHECK_OR_RETURN_ERROR(
        view_byte_span(sizes, strides, element_size, &lowest, &end),
        InvalidArgument,
        "aoti_torch__reinterpret_tensor: the view's extent overflows");
    const auto* first = static_cast<const uint8_t*>(adjusted_data) + lowest;
    const auto* last = static_cast<const uint8_t*>(adjusted_data) + end;
    const uint8_t* begin = nullptr;
    size_t nbytes = 0;
    void* allocation = owning_allocation(self, data_ptr);
    auto cpu_allocation = allocation != nullptr
        ? cpu_allocation_bytes.find(allocation)
        : cpu_allocation_bytes.end();
    void* base = nullptr;
    bool cpu = false;
    if (cpu_allocation != cpu_allocation_bytes.end()) {
      begin = static_cast<const uint8_t*>(allocation);
      nbytes = cpu_allocation->second;
    } else if (
        metal_is_device_pointer(data_ptr) && !metal_is_cpu_memory(data_ptr) &&
        metal_find_memory(data_ptr, &base, &cpu, &nbytes)) {
      // A constant's memory ends where the next constant's starts.
      begin = static_cast<const uint8_t*>(base);
      nbytes = metal_constant_extent(base, nbytes);
    }
    ET_CHECK_OR_RETURN_ERROR(
        begin == nullptr || (begin <= first && last <= begin + nbytes),
        InvalidArgument,
        "aoti_torch__reinterpret_tensor: view at storage_offset=%lld lies "
        "outside the %zu bytes of the memory at %p",
        storage_offset,
        nbytes,
        begin);
  }

  // A view that is not densely packed (e.g. one half of a channels-last tensor
  // chunked along C) cannot be described by a tensor's strides here, so the
  // tensor gets the strides of a packed one.
  void* tensor_data = adjusted_data;
  bool owns_buffer = false;
  bool strided_view = false;
  std::vector<int64_t> view_sizes;
  std::vector<int64_t> view_strides;
  // Whether the new handle took a registration on the view at its address.
  bool registered = false;
  if (!is_packed_strides(sizes, strides)) {
    // The tensor gets contiguous strides either way (a strided view records
    // its real ones separately), and they must fit a tensor's.
    ET_CHECK_OR_RETURN_ERROR(
        fits_tensor(ndim, sizes_ptr, nullptr),
        InvalidArgument,
        "aoti_torch__reinterpret_tensor: a packed copy of the view would "
        "have strides that do not fit a tensor");
    if (metal_is_device_pointer(data_ptr) && !metal_is_cpu_memory(data_ptr)) {
      // Leave the view inside its parent. Kernels generated by inductor index
      // it with the strides they were compiled with and write through it (a cat
      // is filled through such views), so a packed copy would both lose those
      // writes and be indexed past its end. Ops that need dense input take a
      // packed copy when they run (metal_packed_copy_of_strided_view).
      ET_LOG(
          Debug,
          "aoti_torch__reinterpret_tensor: non-packed strides, keeping the view in its parent's buffer");
      // The packed copy ops take of it cannot step backwards.
      ET_CHECK_OR_RETURN_ERROR(
          std::all_of(
              strides.begin(),
              strides.end(),
              [](auto stride) { return stride >= 0; }),
          InvalidArgument,
          "aoti_torch__reinterpret_tensor: a view with negative strides can "
          "only be materialized, not kept in a Metal buffer");
      strided_view = true;
      view_sizes.assign(sizes.begin(), sizes.end());
      view_strides.assign(strides.begin(), strides.end());
    } else {
      ET_LOG(
          Debug,
          "aoti_torch__reinterpret_tensor: non-packed strides, "
          "materializing to packed buffer");
      tensor_data = materialize_packed(
          metal_is_device_pointer(data_ptr),
          adjusted_data,
          sizes,
          strides,
          element_size);
      ET_CHECK_OR_RETURN_ERROR(
          tensor_data != nullptr,
          MemoryAllocationFailed,
          "Failed to materialize non-packed tensor");
      owns_buffer = true;
    }

    // Contiguous strides for the packed layout
    strides = tensor_strides(ndim, sizes_ptr, nullptr);
  }

  std::shared_ptr<Tensor> tensor = make_strided_tensor(
      tensor_data, sizes, strides, dtype_to_scalar_type(dtype));

  if (tensor == nullptr && owns_buffer) {
    metal_deallocate_buffer(tensor_data);
  }
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "Failed to create reinterpreted tensor view");

  if (owns_buffer) {
    // The materialized buffer is a new allocation owned by this tensor
    memory_to_n_tensor[tensor_data] = 1;
  } else {
    // A view that starts where an allocation starts is that allocation.
    auto at_address = memory_to_n_tensor.find(adjusted_data);
    const bool at_allocation =
        at_address != memory_to_n_tensor.end() && at_address->second != NOT_OWN;
    if (adjusted_data != data_ptr && !at_allocation) {
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
            owned ||
                region_can_cover(
                    region, region_nbytes, adjusted_data, tensor->nbytes()),
            InvalidArgument,
            "aoti_torch__reinterpret_tensor: a view of the memory at %p would "
            "take in memory another tensor lies in",
            region);
        ET_CHECK_OR_RETURN_ERROR(
            metal_register_cpu_view(
                adjusted_data, tensor->nbytes(), region, region_nbytes, owned),
            Internal,
            "Failed to wrap the CPU memory of adjusted_data=%p in a Metal buffer",
            adjusted_data);
      }

      add_not_own_handle(adjusted_data);
      registered = true;
    } else {
      // Another handle at the address of `self`, or at the start of the
      // allocation it lies in. If that address is a view, deleting either
      // handle must leave it registered for the other one.
      registered = metal_retain_view(adjusted_data);
      if (!at_allocation) {
        add_not_own_handle(adjusted_data);
      }
    }

    // The new handle keeps the allocation it lives in alive, including when
    // `self` is itself a view. A view at an allocation's start is a handle to
    // that allocation, whatever `self` is.
    hold_allocation(
        tensor.get(),
        adjusted_data,
        at_allocation ? adjusted_data : owning_allocation(self, data_ptr));
  }

  // Only now that nothing can fail: a handle whose registration failed must
  // not be left behind for cleanup to unregister.
  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  if (strided_view) {
    // Non-packed, so not empty, and checked for negative strides above.
    metal_record_strided_view(
        tensor.get(), std::move(view_sizes), std::move(view_strides));
  }
  if (registered) {
    view_handles.insert(tensor.get());
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
  if (data_ptr == nullptr) {
    // Another handle to a tensor with no memory.
    return make_memoryless_tensor(
        ndim, sizes_ptr, strides_ptr, dtype, new_handle);
  }

  // Check if the given memory is in the map
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  // Convert sizes and strides to vectors
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = tensor_strides(ndim, sizes_ptr, strides_ptr);

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
  if (metal_retain_view(data_ptr)) {
    view_handles.insert(tensor.get());
  }
  auto memory = memory_to_n_tensor.find(data_ptr);
  if (memory != memory_to_n_tensor.end() && memory->second == NOT_OWN) {
    add_not_own_handle(data_ptr);
  }

  // The new handle keeps the allocation the original lives in alive.
  hold_allocation(
      tensor.get(), data_ptr, owning_allocation(orig_handle, data_ptr));

  ET_LOG(Debug, "aoti_torch_new_tensor_handle: successful");
  return Error::Ok;
}

// Cleanup function for clearing global state
void cleanup_memory() {
  // Work may still be queued, e.g. after a failed run, and the memory of
  // tensors and constants is about to go.
  auto* stream = getCurrentMetalStream();
  if (stream != nullptr) {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  }

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

  // With every tensor gone nothing is tracked anymore, and a stale entry would
  // make the next model fail to load as soon as its constants land on an
  // address used before.
  memory_to_n_tensor.clear();
  not_own_handles.clear();
  view_owner.clear();
  view_handles.clear();
  cpu_allocation_bytes.clear();

  // Clean up Metal resources
  metal_cleanup_resources();

  ET_LOG(Info, "Cleared all tensors and Metal resources");
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
