/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/module/state_dict_util.h>

namespace executorch {
namespace extension {
namespace training {

runtime::Result<std::map<std::string, executorch::extension::TensorPtr>>
load_state_dict(const runtime::NamedDataMap& data_map) {
  std::map<std::string, executorch::extension::TensorPtr> state_dict;
  auto num_key_res = data_map.get_num_keys();
  if (!num_key_res.ok()) {
    return num_key_res.error();
  }
  for (size_t i = 0; i < num_key_res.get(); i++) {
    // get the key
    auto key_res = data_map.get_key(i);
    if (!key_res.ok()) {
      return key_res.error();
    }

    // get the metadata
    auto metadata_res = data_map.get_tensor_layout(key_res.get());
    if (!metadata_res.ok()) {
      return metadata_res.error();
    }

    // get data blob
    void* data = nullptr;
    static constexpr size_t kMallocAlignment = alignof(std::max_align_t);
    if constexpr (kMallocAlignment < 8) {
      // Skip manually aligning the memory since PyTorch doesn't have dtypes >
      // 8 bytes wide, and I don't expect to ever encounter a platform where
      // malloc aligns to less than 8.
      ET_LOG(
          Error,
          "kMallocAlignment is too small: %zu. Cannot safely create buffer to load tensor. Please open an issue on https://github.com/pytorch/executorch/issues",
          kMallocAlignment);
      return runtime::Error::NotSupported;
    }

    data = malloc(metadata_res->nbytes());
    if (data == nullptr && metadata_res->nbytes() != 0) {
      ET_LOG(Error, "Failed to allocate memory for tensor, malloc failed");
      return runtime::Error::MemoryAllocationFailed;
    }
    auto load_into_error =
        data_map.load_data_into(key_res.get(), data, metadata_res->nbytes());
    if (load_into_error != runtime::Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load data into tensor, likely a malformed .ptd 0x%" PRIx32,
          static_cast<uint32_t>(load_into_error));
      return load_into_error;
    }

    // Get metadata
    std::vector<executorch::aten::SizesType> sizes;
    for (auto x : metadata_res->sizes()) {
      sizes.push_back(x);
    }
    std::vector<executorch::aten::DimOrderType> dim_order;
    for (auto x : metadata_res->dim_order()) {
      dim_order.push_back(x);
    }
    std::vector<executorch::aten::StridesType> strides;
    for (auto stride_index = 0; stride_index < metadata_res->sizes().size();
         stride_index++) {
      if (stride_index == 0) {
        strides.push_back(1);
      } else {
        strides.insert(
            strides.begin(),
            sizes.at(stride_index) * strides.at(stride_index - 1));
      }
    }

    // create tensor
    auto tensor = make_tensor_ptr(
        sizes,
        data,
        dim_order,
        strides,
        metadata_res->scalar_type(),
        exec_aten::TensorShapeDynamism::STATIC,
        [](void* ptr) {
          free(ptr);
          ptr = nullptr;
        });

    // add to state dict
    state_dict.insert({std::string(key_res.get()), std::move(tensor)});
  }

  return state_dict;
}

} // namespace training
} // namespace extension
} // namespace executorch
