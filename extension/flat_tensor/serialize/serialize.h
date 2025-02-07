/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <map>
#include <string>

namespace executorch {
namespace extension {
namespace flat_tensor {

/**
 * Schema version of the .ptd format. Should be kept in sync with serialize.py
 */
constexpr uint32_t kSchemaVersion = 0;

/**
 * Creates a .ptd from the given tensor map.
 *
 * @param path The file path to save the .ptd to.
 * @param tensor_map The map of tensor names to tensors to save.
 * @param tensor_alignment The bytes tensor data should be aligned to.
 * @return An error if the data could not be saved. Error::Ok for success.
 */
ET_EXPERIMENTAL runtime::Error save_ptd(
    const std::string& path,
    const std::map<std::string, executorch::aten::Tensor>& tensor_map,
    const size_t tensor_alignment);

/**
 * Creates a .ptd from the given tensor map.
 *
 * @param out The stream to write the .ptd data to.
 * @param tensor_map The map of tensor names to tensors to save.
 * @param tensor_alignment The bytes tensor data should be aligned to.
 * @return An error if the data could not be saved. Error::Ok for success.
 */
ET_EXPERIMENTAL runtime::Error save_ptd(
    std::ostream& out,
    const std::map<std::string, executorch::aten::Tensor>& tensor_map,
    const size_t tensor_alignment);

} // namespace flat_tensor
} // namespace extension
} // namespace executorch
