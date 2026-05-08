/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Linker stubs for AOTI dtype trampolines that aoti/common_shims_slim
// doesn't define. Required so dlopen of an AOTI .so resolves cleanly
// even when the model never actually uses the unsupported dtype.

#include <cstdint>

extern "C" {

// PyTorch float16 dtype code = c10::ScalarType::Half. Models that
// actually USE float16 will fault inside SlimTensor::check_supportive
// because slim::c10::ScalarType has no Half variant; this stub just
// satisfies the linker.
int32_t aoti_torch_dtype_float16() {
  return 5;
}

}  // extern "C"
