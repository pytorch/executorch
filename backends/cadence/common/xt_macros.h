/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#define XT_KERNEL_CHECK(ctx, out, kernel, ...)            \
  {                                                       \
    const auto ret = kernel(__VA_ARGS__);                 \
    ET_KERNEL_CHECK_MSG(                                  \
        ctx,                                              \
        ret == 0,                                         \
        InvalidArgument,                                  \
        out,                                              \
        "Failed to run kernel: " #kernel "(" #__VA_ARGS__ \
        "). Returned code %d",                            \
        static_cast<int>(ret));                           \
  }
