/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pal/Path.h>

std::string pal::path::GetLibraryName(const std::string& baseName) {
  return baseName + ".dll";
}
