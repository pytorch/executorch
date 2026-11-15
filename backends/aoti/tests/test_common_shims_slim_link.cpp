/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>

#include <gtest/gtest.h>

// A compiled AOTI library calls these shims, but the host never does. This test
// must not call them directly either: a direct call would keep the symbols
// linked even if common_shims_slim stopped being force-linked.
TEST(CommonShimsSlimLinkTest, ShimsAreBindableFromTheHost) {
  constexpr const char* kShims[] = {
      "aoti_torch_check",
      "aoti_torch_device_type_cpu",
      "aoti_torch_device_type_cuda",
      "aoti_torch_dtype_bfloat16",
      "aoti_torch_dtype_bool",
      "aoti_torch_dtype_float16",
      "aoti_torch_dtype_float32",
      "aoti_torch_dtype_float64",
      "aoti_torch_dtype_int16",
      "aoti_torch_dtype_int32",
      "aoti_torch_dtype_int64",
      "aoti_torch_dtype_int8",
      "aoti_torch_dtype_uint8",
      "aoti_torch_get_data_ptr",
      "aoti_torch_get_device_index",
      "aoti_torch_get_device_type",
      "aoti_torch_get_dim",
      "aoti_torch_get_dtype",
      "aoti_torch_get_numel",
      "aoti_torch_get_size",
      "aoti_torch_get_sizes",
      "aoti_torch_get_storage_offset",
      "aoti_torch_get_storage_size",
      "aoti_torch_get_strides",
      "aoti_torch_grad_mode_is_enabled",
      "aoti_torch_grad_mode_set_enabled",
      "aoti_torch_is_defined",
      "aoti_torch_layout_strided",
  };
  for (const char* name : kShims) {
    EXPECT_NE(dlsym(RTLD_DEFAULT, name), nullptr) << name;
  }
}
