#pragma once

#include <vector>

#include <executorch/backends/aoti/slim/c10/core/MemoryFormat.h>
#include <executorch/backends/aoti/slim/core/Storage.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace executorch::backends::aoti::slim {
inline void SlimTensor::empty_tensor_restride(
    executorch::backends::aoti::slim::c10::MemoryFormat memory_format) {
#ifdef DEBUG
  STANDALONE_INTERNAL_ASSERT(
      compute_numel() == numel_,
      "If you are seeing this error, that means empty_tensor_restride was "
      "called before setting correct numel");
#endif
  switch (memory_format) {
    case executorch::backends::aoti::slim::c10::MemoryFormat::Contiguous: {
      // dim_ is a virtual call, don't repeat it
      const auto dim_ = dim();
      sizes_and_strides_.resize(dim_);
      if (dim_ > 0) {
        bool overflowed = false;
        const auto last_idx = dim_ - 1;
        sizes_and_strides_.stride_at_unchecked(last_idx) = 1;
        for (int64_t i = static_cast<int64_t>(last_idx) - 1; i >= 0; --i) {
          overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
              sizes_and_strides_.stride_at_unchecked(i + 1),
              std::max<int64_t>(sizes_and_strides_.size_at_unchecked(i + 1), 1),
              std::addressof(sizes_and_strides_.stride_at_unchecked(i)));
        }
        STANDALONE_CHECK(!overflowed, "Stride calculation overflowed");
      }
      break;
    }
    case executorch::backends::aoti::slim::c10::MemoryFormat::ChannelsLast: {
      STANDALONE_CHECK(
          dim() == 4, "required rank 4 tensor to use channels_last format");
      set_sizes_and_strides(sizes(), get_channels_last_strides_2d(sizes()));
      break;
    }
    case executorch::backends::aoti::slim::c10::MemoryFormat::ChannelsLast3d: {
      STANDALONE_CHECK(
          dim() == 5, "required rank 5 tensor to use channels_last_3d format");
      set_sizes_and_strides(sizes(), get_channels_last_strides_3d(sizes()));
      break;
    }
    case executorch::backends::aoti::slim::c10::MemoryFormat::Preserve:
      STANDALONE_CHECK(false, "unsupported memory format ", memory_format);
      // Cleaning warning messages, no need to break as STANDALONE_CHECK(false)
      // terminates flow.
      // break;
    case executorch::backends::aoti::slim::c10::MemoryFormat::NumOptions:
      STANDALONE_INTERNAL_ASSERT(
          false, "invalid memory format ", memory_format);
  }
  // recompute contiguous flag, as currently NHWC/NCHW flags are not mutually
  // exclusive see #24090
  refresh_contiguous();
}

inline void _resize_bytes(
    MaybeOwningStorage* storage,
    size_t new_size_bytes,
    size_t storage_offset_in_bytes) {
  STANDALONE_CHECK(
      storage->is_resizable(),
      "Trying to resize storage that is not resizable");

  void* new_data = nullptr;
  const c10::Device& device = storage->device();
  if (new_size_bytes > 0) {
    if (device.is_cpu()) {
      new_data =
          DeviceTraits<c10::DeviceType::CPU>::allocate(new_size_bytes, device);
    } else if (device.is_cuda()) {
      new_data =
          DeviceTraits<c10::DeviceType::CUDA>::allocate(new_size_bytes, device);
    }
  }

  void* old_data = storage->data();
  const size_t old_capacity = storage->nbytes();
  const size_t copy_capacity = std::min(new_size_bytes, old_capacity);
  if (old_data != nullptr && copy_capacity > 0) {
    if (device.is_cpu()) {
      DeviceTraits<c10::DeviceType::CPU>::memcpy(
          static_cast<char*>(new_data) + storage_offset_in_bytes,
          static_cast<char*>(old_data) + storage_offset_in_bytes,
          copy_capacity,
          device,
          device);
    } else if (device.is_cuda()) {
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          static_cast<char*>(new_data) + storage_offset_in_bytes,
          static_cast<char*>(old_data) + storage_offset_in_bytes,
          copy_capacity,
          device,
          device);
    }
  }

  storage->free_data();
  storage->set_data_ptr_noswap(new_data);
  storage->set_nbytes(new_size_bytes);
}

inline void _maybe_resize_storage(SlimTensor* self, int64_t new_size_bytes) {
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->storage();
  if (!storage) {
    Storage new_storage(new MaybeOwningStorage(self->device(), new_size_bytes));
    self->set_storage(std::move(new_storage));
  } else if (new_size_bytes > static_cast<int64_t>(self->nbytes())) {
    _resize_bytes(
        storage.get(),
        new_size_bytes,
        self->storage_offset() * self->itemsize());
  }
}

inline SlimTensor* _resize_impl_(
    SlimTensor* self,
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    std::optional<executorch::backends::aoti::slim::c10::IntArrayRef> strides,
    bool resize_storage) {
  if (self->sizes() == sizes &&
      (!strides || self->strides() == strides.value())) {
    return self;
  }

  const auto itemsize = self->itemsize();
  const auto storage_offset = self->storage_offset();
  int64_t storage_size = 1;
  if (strides) {
    self->set_sizes_and_strides(sizes, *strides);
    storage_size =
        compute_storage_nbytes(sizes, *strides, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(sizes);
    storage_size =
        compute_storage_nbytes_contiguous(sizes, itemsize, storage_offset);
  }

  if (resize_storage) {
    _maybe_resize_storage(self, storage_size);
  }

  return self;
}

inline SlimTensor SlimTensor::resize_(
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  _resize_impl_(this, sizes, /*stride=*/std::nullopt, true);

  if (optional_memory_format.has_value()) {
    executorch::backends::aoti::slim::c10::MemoryFormat memory_format =
        static_cast<executorch::backends::aoti::slim::c10::MemoryFormat>(
            optional_memory_format.value());
    STANDALONE_CHECK(
        memory_format !=
            executorch::backends::aoti::slim::c10::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    this->empty_tensor_restride(memory_format);
  }
  return *this;
}

} // namespace executorch::backends::aoti::slim
