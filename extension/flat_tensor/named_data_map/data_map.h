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

#include <utility>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace flat_tensor_flatbuffer {
struct FlatTensor;
} // namespace flat_tensor_flatbuffer

namespace executorch {
namespace extension {

class DataMap final : public executorch::runtime::NamedDataMap {
 public:
  static executorch::runtime::Result<DataMap> load(
      executorch::runtime::DataLoader* loader);

  ET_NODISCARD
  executorch::runtime::Result<const executorch::runtime::TensorLayout>
  get_metadata(const char* key) const override;
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      const char* key) const override;
  ET_NODISCARD executorch::runtime::Result<size_t>
  load_data_into(const char* key, void* buffer, size_t size) const override;

  ET_NODISCARD executorch::runtime::Result<size_t> get_num_keys()
      const override;
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      size_t index) const override;

  DataMap(DataMap&&) noexcept = default;
  ~DataMap() override;

 private:
  DataMap(
      executorch::runtime::DataLoader* loader,
      size_t segment_base_offset,
      executorch::runtime::FreeableBuffer&& flat_tensor_data,
      const flat_tensor_flatbuffer::FlatTensor* flat_tensor,
      executorch::runtime::FreeableBuffer&& data_ro)
      : _loader(loader),
        _segment_base_offset(segment_base_offset),
        _flat_tensor_data(std::move(flat_tensor_data)),
        _flat_tensor(flat_tensor),
        _data_ro(std::move(data_ro)){};

  // Not copyable or assignable.
  DataMap(const DataMap& rhs) = delete;
  DataMap& operator=(DataMap&& rhs) noexcept = delete;
  DataMap& operator=(const DataMap& rhs) = delete;

  // Data loader used to load segment data.
  executorch::runtime::DataLoader* _loader;

  // Segment base offset.
  size_t _segment_base_offset;

  // Serialized flat_tensor data.
  executorch::runtime::FreeableBuffer _flat_tensor_data;

  // Flatbuffer representation of the flat_tensor.
  const flat_tensor_flatbuffer::FlatTensor* _flat_tensor;

  // Loaded read-only tensor data.
  executorch::runtime::FreeableBuffer _data_ro;
};

} // namespace extension
} // namespace executorch
