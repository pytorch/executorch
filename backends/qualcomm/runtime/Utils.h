/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <string>

namespace torch {
namespace executor {
namespace qnn {
// Create Directory
void CreateDirectory(const std::string& path);

} // namespace qnn
} // namespace executor
} // namespace torch
