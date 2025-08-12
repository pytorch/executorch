/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/compiler.h>

#include <map>
#include <string>

namespace executorch {
namespace extension {
namespace training {

/**
 * Generate a map of string to tensor.
 *
 * @param data The NamedDataMap to load the tensors and names from.
 * @return A result containing a map of tensor names to tensors if
 *   successful, an error otherwise.
 */
ET_EXPERIMENTAL
runtime::Result<std::map<std::string, executorch::extension::TensorPtr>>
load_state_dict(const runtime::NamedDataMap& data);

} // namespace training
} // namespace extension
} // namespace executorch
