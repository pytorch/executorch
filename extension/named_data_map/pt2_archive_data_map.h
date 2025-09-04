/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/compiler.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "miniz.h"

namespace executorch {
namespace extension {

/**
 * A NamedDataMap implementation for PT2Archive-serialized data.
 */
class PT2ArchiveDataMap final
    : public executorch::ET_RUNTIME_NAMESPACE::NamedDataMap {
 public:
  /**
   * Creates a new PT2ArchiveDataMap that wraps PT2Archive data.
   *`
   * @param[in] pt2_archive_file_path The path to the PT2Archive file.
   */
  static executorch::runtime::Result<PT2ArchiveDataMap> load(
      const std::string& pt2_archive_file_path);

  /**
   * Retrieve the tensor_layout for the specified key.
   *
   * @param[in] key The name of the tensor to get metadata on.
   *
   * @return Error::NotFound if the key is not present.
   */
  ET_NODISCARD
  executorch::runtime::Result<
      const executorch::ET_RUNTIME_NAMESPACE::TensorLayout>
  get_tensor_layout(executorch::aten::string_view key) const override;

  /**
   * Retrieve read-only data for the specified key.
   *
   * @param[in] key The name of the tensor to get data on.
   *
   * @return error if the key is not present or data cannot be loaded.
   */
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      executorch::aten::string_view key) const override;

  /**
   * Loads the data of the specified tensor into the provided buffer.
   *
   * @param[in] key The name of the tensor to get the data of.
   * @param[in] buffer The buffer to load data into. Must point to at least
   * `size` bytes of memory.
   * @param[in] size The number of bytes to load.
   *
   * @returns an Error indicating if the load was successful.
   */
  ET_NODISCARD executorch::runtime::Error load_data_into(
      executorch::aten::string_view key,
      void* buffer,
      size_t size) const override;

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD executorch::runtime::Result<uint32_t> get_num_keys()
      const override;

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      uint32_t index) const override;

  PT2ArchiveDataMap(PT2ArchiveDataMap&&) noexcept = default;

  ~PT2ArchiveDataMap() override;

 private:
  // Used to back the TensorLayout class. This allows us to free the json
  // blobs, instead of parsing them on each get_tensor_layout call.
  struct ConcreteTensorLayout {
    std::vector<int32_t> sizes;
    std::vector<uint8_t> dim_order;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    executorch::aten::ScalarType scalar_type;

    executorch::runtime::Result<
        const executorch::ET_RUNTIME_NAMESPACE::TensorLayout>
    create_tensor_layout() const {
      return executorch::ET_RUNTIME_NAMESPACE::TensorLayout::create(
          executorch::runtime::Span<const int32_t>(sizes.data(), sizes.size()),
          executorch::runtime::Span<const uint8_t>(
              dim_order.data(), dim_order.size()),
          scalar_type);
    }
  };

  static executorch::runtime::Error parse_json(
      std::unique_ptr<mz_zip_archive>& zip_archive,
      const std::string& filename,
      std::unordered_map<std::string, std::string>& tensor_name_to_path,
      std::unordered_map<std::string, ConcreteTensorLayout>&
          tensor_name_to_layout);

  PT2ArchiveDataMap(
      std::unique_ptr<mz_zip_archive> zip_archive,
      std::unique_ptr<executorch::runtime::DataLoader> loader,
      std::string archive_name,
      std::unordered_map<std::string, ConcreteTensorLayout>
          tensor_name_to_layout,
      std::unordered_map<std::string, std::string> tensor_name_to_path)
      : zip_archive_(std::move(zip_archive)),
        loader_(std::move(loader)),
        archive_name_(std::move(archive_name)),
        tensor_name_to_layout_(std::move(tensor_name_to_layout)),
        tensor_name_to_path_(std::move(tensor_name_to_path)) {}

  // Not copyable or assignable.
  PT2ArchiveDataMap(const PT2ArchiveDataMap& rhs) = delete;
  PT2ArchiveDataMap& operator=(PT2ArchiveDataMap&& rhs) noexcept = delete;
  PT2ArchiveDataMap& operator=(const PT2ArchiveDataMap& rhs) = delete;

  // Open zip archive.
  std::unique_ptr<mz_zip_archive> zip_archive_;
  // Data loader, used to load weights.
  std::unique_ptr<executorch::runtime::DataLoader> loader_;
  // Archive name without extensions, used to find weights in the archive.
  std::string archive_name_;

  // Weight data from JSON files.
  std::unordered_map<std::string, ConcreteTensorLayout> tensor_name_to_layout_;
  std::unordered_map<std::string, std::string> tensor_name_to_path_;
};

} // namespace extension
} // namespace executorch
