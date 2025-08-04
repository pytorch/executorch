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
namespace executorch_flatbuffer {
struct NamedData;
struct DataSegment;
} // namespace executorch_flatbuffer

namespace flatbuffers {
template <typename T>
struct Offset;
} // namespace flatbuffers

// @lint-ignore CLANGTIDY facebook-modularize-issue-check
#if EXECUTORCH_INTERNAL_FLATBUFFERS == 1
// TODO(T216992074): update internal flatbuffers (v1.12) to match OSS (v24.3.5).
namespace flatbuffers {
template <typename T>
class Vector;
using FlatbufferNamedData =
    flatbuffers::Vector<flatbuffers::Offset<executorch_flatbuffer::NamedData>>;
using FlatbufferDataSegment = flatbuffers::Vector<
    flatbuffers::Offset<executorch_flatbuffer::DataSegment>>;
} // namespace flatbuffers
#else
namespace flatbuffers {
template <typename T, typename SizeT>
class Vector;
using FlatbufferNamedData = flatbuffers::
    Vector<flatbuffers::Offset<executorch_flatbuffer::NamedData>, uint32_t>;
using FlatbufferDataSegment = flatbuffers::
    Vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>, uint32_t>;
} // namespace flatbuffers
#endif

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace internal {

/**
 * A NamedDataMap implementation for Flatbuffer-serialized named data
 * originating from a PTE file.
 */
class PteDataMap final : public NamedDataMap {
 public:
  /**
   * Creates a new DataMap that wraps named_data from the PTE file.
   *
   * @param[in] loader The DataLoader that accesses the PTE file.
   * Note: the loader must outlive the PteDataMap instance.
   * @param[in] segment_base_offset The offset to the first segment in the PTE
   * file, in bytes.
   * @param[in] named_data The named_data from the PTE file. Note: the pointer
   * passed here must outlive the PteDataMap instance.
   * @param[in] segments The segments from the PTE file. Note: the pointer
   * passed here must outlive the PteDataMap instance.
   */
  static Result<PteDataMap> create(
      DataLoader* loader,
      size_t segment_base_offset,
      const flatbuffers::FlatbufferNamedData* named_data,
      const flatbuffers::FlatbufferDataSegment* segments);

  /**
   * The PteDataMap currently only handles opaque data that does not contain
   * tensor-specific metadata.
   */
  ET_NODISCARD
  Result<const TensorLayout> get_tensor_layout(
      ET_UNUSED executorch::aten::string_view key) const override {
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
  Result<FreeableBuffer> get_data(
      executorch::aten::string_view key) const override;

  /**
   * The PteDataMap currently does not implement load_into.
   */
  ET_NODISCARD Error load_data_into(
      ET_UNUSED executorch::aten::string_view key,
      ET_UNUSED void* buffer,
      ET_UNUSED size_t size) const override {
    return Error::NotImplemented;
  }

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD Result<uint32_t> get_num_keys() const override;

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD Result<const char*> get_key(uint32_t index) const override;

  // Moveable, to be compatible with Result.
  PteDataMap(PteDataMap&&) noexcept = default;
  ~PteDataMap() override = default;

 private:
  PteDataMap(
      DataLoader* loader,
      size_t segment_base_offset,
      const flatbuffers::FlatbufferNamedData* named_data,
      const flatbuffers::FlatbufferDataSegment* segments)
      : loader_(loader),
        segment_base_offset_(segment_base_offset),
        named_data_(named_data),
        segments_(segments) {}

  // Not copyable or assignable.
  PteDataMap(const PteDataMap& rhs) = delete;
  PteDataMap& operator=(PteDataMap&& rhs) noexcept = delete;
  PteDataMap& operator=(const PteDataMap& rhs) = delete;

  // Data loader, used to load segment data.
  DataLoader* loader_;

  // The offset to the first segment in the PTE file, in bytes.
  size_t segment_base_offset_;

  // Named data, containing name and segment index.
  const flatbuffers::FlatbufferNamedData* named_data_;

  // Segments, to retrieve offset and size for the loader.
  const flatbuffers::FlatbufferDataSegment* segments_;
};

} // namespace internal
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
