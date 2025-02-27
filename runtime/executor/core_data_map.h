/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/named_data_map.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace flatbuffers {
template <typename T>
class Vector;
template <typename T>
struct Offset;
} // namespace flatbuffers

namespace executorch_flatbuffer {
struct NamedData;
struct DataSegment;
} // namespace executorch_flatbuffer

namespace executorch {
namespace runtime {

/**
 * A NamedDataMap implementation for Flatbuffer-serialized named data
 * originating from a PTE file.
 */
class CoreDataMap final : public executorch::runtime::NamedDataMap {
 public:
  /**
   * Creates a new DataMap that wraps named_data from the PTE file.
   *
   * @param[in] loader The DataLoader that accesses the PTE file.
   * Note: the loader must outlive the CoreDataMap instance.
   * @param[in] segment_base_offset The offset to the first segment in the PTE
   * file, in bytes.
   * @param[in] named_data The named_data from the PTE file. Note: the pointer
   * passed here must outlive the CoreDataMap instance.
   * @param[in] segments The segments from the PTE file. Note: the pointer
   * passed here must outlive the CoreDataMap instance.
   */
  static executorch::runtime::Result<CoreDataMap> load(
      executorch::runtime::DataLoader* loader,
      size_t segment_base_offset,
      const flatbuffers::Vector<
          flatbuffers::Offset<executorch_flatbuffer::NamedData>>* named_data,
      const flatbuffers::Vector<
          flatbuffers::Offset<executorch_flatbuffer::DataSegment>>* segments);

  /**
   * The CoreDataMap currently only handles opaque data that does not contain
   * tensor-specific metadata.
   */
  ET_NODISCARD
  executorch::runtime::Result<const executorch::runtime::TensorLayout>
  get_metadata(ET_UNUSED const char* key) const override {
    return Error::NotImplemented;
  }

  /**
   * Retrieve read-only data for the specified key.
   *
   * @param[in] key The name of the blob to get data on.
   *
   * @return error if the key is not present or data cannot be loaded.
   */
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      const char* key) const override;

  /**
   * The CoreDataMap currently does not implement load_into.
   */
  ET_NODISCARD executorch::runtime::Error load_data_into(
      ET_UNUSED const char* key,
      ET_UNUSED void* buffer,
      ET_UNUSED size_t size) const override {
    return Error::NotImplemented;
  }

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD executorch::runtime::Result<size_t> get_num_keys()
      const override;

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      size_t index) const override;

  // Moveable, to be compatible with Result.
  CoreDataMap(CoreDataMap&&) noexcept = default;
  ~CoreDataMap() override = default;

 private:
  CoreDataMap(
      executorch::runtime::DataLoader* loader,
      size_t segment_base_offset,
      const flatbuffers::Vector<
          flatbuffers::Offset<executorch_flatbuffer::NamedData>>* named_data,
      const flatbuffers::Vector<
          flatbuffers::Offset<executorch_flatbuffer::DataSegment>>* segments)
      : loader_(loader),
        segment_base_offset_(segment_base_offset),
        named_data_(named_data),
        segments_(segments) {}

  // Not copyable or assignable.
  CoreDataMap(const CoreDataMap& rhs) = delete;
  CoreDataMap& operator=(CoreDataMap&& rhs) noexcept = delete;
  CoreDataMap& operator=(const CoreDataMap& rhs) = delete;

  // Data loader, used to load segment data.
  executorch::runtime::DataLoader* loader_;

  // Segment base offset.
  size_t segment_base_offset_;

  // Named data, containing name and segment index.
  const flatbuffers::Vector<
      flatbuffers::Offset<executorch_flatbuffer::NamedData>>* named_data_;

  // Segments, to retrieve offset and size for the loader.
  const flatbuffers::Vector<
      flatbuffers::Offset<executorch_flatbuffer::DataSegment>>* segments_;
};

} // namespace runtime
} // namespace executorch
