/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/compiler.h>

#include <unordered_map>
#include <utility>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace flat_tensor {
struct FlatTensor;
} // namespace flat_tensor

namespace executorch {
namespace extension {

class DataMap final : public executorch::runtime::NamedDataMap {
 public:
  static executorch::runtime::Result<DataMap> load(
      executorch::runtime::DataLoader* loader);

  ET_NODISCARD
  executorch::runtime::Result<const executorch::runtime::TensorLayout>
  get_metadata(const char* fqn) const override;
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      const char* fqn) const override;
  ET_NODISCARD runtime::Error
  load_data_into(const char* fqn, size_t size, void* buffer) const override;

  ET_NODISCARD executorch::runtime::Result<int> get_num_keys() const override;
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      int index) const override;

  DataMap(DataMap&&) noexcept = default;
  ~DataMap() override;

 private:
  DataMap(
      executorch::runtime::FreeableBuffer&& flat_tensor_data,
      std::unordered_map<
          std::string,
          std::tuple<int, int, executorch::runtime::TensorLayout>>
          name_to_tensor,
      executorch::runtime::FreeableBuffer&& data_ro)
      : _flat_tensor_data(std::move(flat_tensor_data)),
        _name_to_tensor(std::move(name_to_tensor)),
        _data_ro(std::move(data_ro)) {}

  // Not copyable or assignable.
  DataMap(const DataMap& rhs) = delete;
  DataMap& operator=(DataMap&& rhs) noexcept = delete;
  DataMap& operator=(const DataMap& rhs) = delete;

  // FlatTensor flatbuffer data. Contains the data backing up
  // TensorLayout information in the _name_to_tensor map; must outlive it.
  executorch::runtime::FreeableBuffer _flat_tensor_data;

  // Map of name to {segment index, offset, TensorLayout}.
  std::unordered_map<
      std::string,
      std::tuple<int, int, executorch::runtime::TensorLayout>>
      _name_to_tensor;

  // Raw, read-only tensor data.
  executorch::runtime::FreeableBuffer _data_ro;
};

} // namespace extension
} // namespace executorch
