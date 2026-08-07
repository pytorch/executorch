/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch::backends::webgpu {

inline int required_device_failure_exit_code(bool required) {
  return required ? 1 : 0;
}

} // namespace executorch::backends::webgpu
