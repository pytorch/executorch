/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace internal {

/**
 * A NamedDataMap implementation that wraps other NamedDataMaps.
 */
class MergedDataMap final : public NamedDataMap {
 public:
  /**
   * Creates a new NamedDataMap that wraps two other data maps.
   *
   * @param[in] first The first NamedDataMap to merge.
   * @param[in] second The second NamedDataMap to merge.
   * Note: the data maps must outlive the MergedDataMap instance.
   */
  static Result<MergedDataMap> load(
      const NamedDataMap* first,
      const NamedDataMap* second) {
    ET_CHECK_OR_RETURN_ERROR(
        first != nullptr && second != nullptr,
        InvalidArgument,
        "Input data map is null.");

    // Check for duplicate keys.
    for (uint32_t k = 0; k < first->get_num_keys().get(); k++) {
      const auto key = first->get_key(k).get();
      const auto error = second->get_tensor_layout(key).error();
      // TODO(lfq): add API to check if key exists.
      ET_CHECK_OR_RETURN_ERROR(
          error == Error::NotFound || error == Error::NotImplemented,
          InvalidArgument,
          "Duplicate key %s.",
          key);
    }
    return MergedDataMap(first, second);
  }

  /**
   * Retrieve the tensor_layout for the specified key.
   *
   * @param[in] key The name of the tensor to get metadata on.
   *
   * @return Error::NotFound if the key is not present.
   */
  ET_NODISCARD
  Result<const TensorLayout> get_tensor_layout(
      std::string_view key) const override {
    auto layout = first_->get_tensor_layout(key);
    if (layout.ok()) {
      return layout.get();
    }
    if (layout.error() != Error::NotFound) {
      return layout.error();
    }
    return second_->get_tensor_layout(key);
  }

  /**
   * Retrieve read-only data for the specified key.
   *
   * @param[in] key The name of the tensor to get data on.
   *
   * @return error if the key is not present or data cannot be loaded.
   */
  ET_NODISCARD
  Result<FreeableBuffer> get_data(std::string_view key) const override {
    auto data = first_->get_data(key);
    if (data.error() != Error::NotFound) {
      return data;
    }
    return second_->get_data(key);
  }

  /**
   * Loads the data of the specified tensor into the provided buffer.
   * Not used in the MergedDataMap.
   *
   * @param[in] key The name of the tensor to get the data of.
   * @param[in] buffer The buffer to load data into. Must point to at least
   * `size` bytes of memory.
   * @param[in] size The number of bytes to load.
   *
   * @returns an Error indicating if the load was successful.
   */
  ET_NODISCARD Error load_data_into(
      ET_UNUSED std::string_view key,
      ET_UNUSED void* buffer,
      ET_UNUSED size_t size) const override {
    return Error::NotImplemented;
  }

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD Result<uint32_t> get_num_keys() const override {
    return first_->get_num_keys().get() + second_->get_num_keys().get();
  }

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD Result<const char*> get_key(uint32_t index) const override {
    uint32_t total_num_keys = get_num_keys().get();
    ET_CHECK_OR_RETURN_ERROR(
        index < total_num_keys,
        InvalidArgument,
        "Index %" PRIu32 " out of range of size %" PRIu32,
        index,
        total_num_keys);

    if (index < first_->get_num_keys().get()) {
      return first_->get_key(index);
    } else {
      return second_->get_key(index - first_->get_num_keys().get());
    }
  }

  ET_NODISCARD Error replace_data(
      Span<const Data> data,
      MemoryAllocator* temp_allocator) const override {
    if (data.empty()) {
      return Error::Ok;
    }
    const NamedDataMap* target = nullptr;
    for (const auto& replacement : data) {
      const NamedDataMap* owner = nullptr;
      const NamedDataMap* maps[] = {first_, second_};
      for (const NamedDataMap* candidate : maps) {
        auto num_keys = candidate->get_num_keys();
        if (!num_keys.ok()) {
          return num_keys.error();
        }
        for (uint32_t i = 0; i < num_keys.get(); ++i) {
          auto key = candidate->get_key(i);
          if (!key.ok()) {
            return key.error();
          }
          if (replacement.key == key.get()) {
            owner = candidate;
            break;
          }
        }
        if (owner != nullptr) {
          break;
        }
      }
      ET_CHECK_OR_RETURN_ERROR(
          owner != nullptr,
          NotFound,
          "Named data key %.*s was not found",
          static_cast<int>(replacement.key.size()),
          replacement.key.data());
      ET_CHECK_OR_RETURN_ERROR(
          target == nullptr || target == owner,
          NotSupported,
          "A replacement cannot span multiple named data sources");
      target = owner;
    }
    return target->replace_data(data, temp_allocator);
  }

  MergedDataMap(MergedDataMap&&) noexcept = default;

  ~MergedDataMap() override = default;

 private:
  MergedDataMap(const NamedDataMap* first, const NamedDataMap* second)
      : first_{first}, second_{second} {}

  // Not copyable or assignable.
  MergedDataMap(const MergedDataMap& rhs) = delete;
  MergedDataMap& operator=(MergedDataMap&& rhs) noexcept = delete;
  MergedDataMap& operator=(const MergedDataMap& rhs) = delete;

  const NamedDataMap* first_;
  const NamedDataMap* second_;
};

} // namespace internal
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
