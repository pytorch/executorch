// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/KernelProvider.h>

#include <algorithm>
#include <cstring>

namespace executorch::backends::cpu {
using namespace executorch::runtime;

bool BufferRequirements::valid() const {
  return alignment >= kBufferAlignment && !(alignment & (alignment - 1));
}

void BufferRequirements::merge(const BufferRequirements& other) {
  readable_bytes = std::max(readable_bytes, other.readable_bytes);
  writable_bytes = std::max(writable_bytes, other.writable_bytes);
  alignment = std::max(alignment, other.alignment);
}

Result<BufferRequirements> tensor_requirements(
    size_t bytes,
    bool writable,
    size_t alignment,
    size_t readable_tail) {
  readable_tail = std::max(readable_tail, kReadableTail);
  BufferRequirements requirements{0, writable ? bytes : 0, alignment};
  ET_CHECK_OR_RETURN_ERROR(
      requirements.valid() && bytes <= SIZE_MAX - readable_tail,
      InvalidArgument,
      "Invalid CPU buffer requirement: bytes=%zu alignment=%zu tail=%zu",
      bytes,
      alignment,
      readable_tail);
  requirements.readable_bytes = bytes + readable_tail;
  return requirements;
}

bool Buffer::accepts(const BufferRequirements& requirements) const {
  const size_t extent = std::max(readable_bytes, writable_bytes);
  return requirements.valid() && data && alignment >= requirements.alignment &&
      alignment % requirements.alignment == 0 &&
      reinterpret_cast<uintptr_t>(data) % requirements.alignment == 0 &&
      reinterpret_cast<uintptr_t>(data) <= UINTPTR_MAX - extent &&
      readable_bytes >= requirements.readable_bytes &&
      writable_bytes >= requirements.writable_bytes;
}

bool Buffer::accepts(size_t bytes, bool writable) const {
  auto requirements = tensor_requirements(bytes, writable);
  return requirements.ok() && accepts(requirements.get());
}

Result<Buffer> allocate_buffer(
    MemoryAllocator& allocator,
    const BufferRequirements& requirements) {
  ET_CHECK_OR_RETURN_ERROR(
      requirements.valid(),
      InvalidArgument,
      "Invalid CPU allocation alignment: %zu",
      requirements.alignment);
  const size_t capacity =
      std::max(requirements.readable_bytes, requirements.writable_bytes);
  auto* data =
      allocator.allocate(std::max(capacity, size_t{1}), requirements.alignment);
  ET_CHECK_OR_RETURN_ERROR(
      data, MemoryAllocationFailed, "CPU allocation failed: %zu", capacity);
  std::memset(data, 0, capacity);
  return Buffer{
      data, capacity, requirements.writable_bytes, requirements.alignment};
}

Result<Buffer> allocate_buffer(MemoryAllocator& allocator, size_t bytes) {
  auto requirements = tensor_requirements(bytes, true);
  if (!requirements.ok()) {
    return requirements.error();
  }
  return allocate_buffer(allocator, requirements.get());
}

Result<size_t> tensor_bytes(const ptn::Value& value) {
  ET_CHECK_OR_RETURN_ERROR(
      value.is_tensor(),
      NotSupported,
      "CPU static delegate requires a tensor: %s",
      value.name.c_str());
  const auto& meta = value.tensor_meta();
  ET_CHECK_OR_RETURN_ERROR(
      meta.dtype == ptn::ScalarType::Float && meta.is_contiguous() &&
          meta.sizes.size() <= 16,
      NotSupported,
      "CPU static delegate requires contiguous FP32 rank <= 16: %s",
      value.name.c_str());
  size_t elements = 1;
  for (auto dimension = meta.sizes.rbegin(); dimension != meta.sizes.rend();
       ++dimension) {
    const auto size = *dimension;
    ET_CHECK_OR_RETURN_ERROR(
        size >= 0 && size <= INT32_MAX &&
            (size == 0 || elements <= INT32_MAX / static_cast<size_t>(size)),
        NotSupported,
        "CPU tensor size exceeds int32: %s",
        value.name.c_str());
    elements *= size;
  }
  return elements * sizeof(float);
}

Result<StorageRequirements> KernelImplementation::requirements(
    const KernelRegion& region,
    const ptn::Graph& graph,
    const ExecutionContext&) const {
  StorageRequirements result;
  result.values.reserve(region.inputs.size() + region.outputs.size());
  for (auto* ids : {&region.inputs, &region.outputs}) {
    for (auto id : *ids) {
      auto bytes = tensor_bytes(graph.value(id));
      if (!bytes.ok()) {
        return bytes.error();
      }
      auto buffer = tensor_requirements(bytes.get(), ids == &region.outputs);
      if (!buffer.ok()) {
        return buffer.error();
      }
      result.values.push_back({id, buffer.get()});
    }
  }
  return result;
}

Error RuntimeConfiguration::apply_options(const BackendInitContext& context) {
  auto count = context.get_runtime_spec<int>(kPreferenceCountOption.data());
  auto provider =
      context.get_runtime_spec<const char*>(kPreferredProviderOption.data());
  auto implementation = context.get_runtime_spec<const char*>(
      kPreferredImplementationOption.data());
  for (const auto* value : {&provider, &implementation}) {
    if (!value->ok() && value->error() != Error::NotFound) {
      return value->error();
    }
  }
  if (count.ok()) {
    ET_CHECK_OR_RETURN_ERROR(
        !provider.ok() && !implementation.ok() && count.get() >= 0 &&
            static_cast<size_t>(count.get()) <= context.runtime_specs().size(),
        InvalidArgument,
        "CPU preference_count requires indexed preferences without single-pair options");
    preferences.clear();
    for (int index = 0; index < count.get(); ++index) {
      const auto suffix = "_" + std::to_string(index);
      const auto provider_key = std::string(kPreferredProviderOption) + suffix;
      const auto implementation_key =
          std::string(kPreferredImplementationOption) + suffix;
      auto name = context.get_runtime_spec<const char*>(provider_key.c_str());
      auto kernel =
          context.get_runtime_spec<const char*>(implementation_key.c_str());
      ET_CHECK_OR_RETURN_ERROR(
          name.ok() && name.get()[0],
          InvalidArgument,
          "CPU preference %d requires a nonempty provider string",
          index);
      if (!kernel.ok() && kernel.error() != Error::NotFound) {
        return kernel.error();
      }
      preferences.push_back({name.get(), kernel.ok() ? kernel.get() : ""});
    }
  } else if (count.error() != Error::NotFound) {
    return count.error();
  } else if (provider.ok() || implementation.ok()) {
    auto preference =
        preferences.size() == 1 ? preferences.front() : KernelPreference{};
    if (provider.ok()) {
      preference.provider = provider.get();
    }
    if (implementation.ok()) {
      preference.implementation = implementation.get();
    }
    preferences.clear();
    if (!preference.provider.empty() || !preference.implementation.empty()) {
      preferences.push_back(std::move(preference));
    }
  }
  auto requested_force = context.get_runtime_spec<bool>(kForceOption.data());
  if (requested_force.ok()) {
    force = requested_force.get();
  } else if (requested_force.error() != Error::NotFound) {
    return requested_force.error();
  }
  return Error::Ok;
}
} // namespace executorch::backends::cpu
