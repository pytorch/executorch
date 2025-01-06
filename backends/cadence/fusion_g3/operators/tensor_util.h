/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/
 
 
#pragma once
 
#define XT_KERNEL_CHECK(ctx, out, kernel, ...) \
  const auto ret = kernel(__VA_ARGS__);        \
  ET_KERNEL_CHECK_MSG(                         \
      ctx,                                     \
      ret == 0,                                \
      InvalidArgument,                         \
      out,                                     \
      "Failed to run kernel: " #kernel "(" #__VA_ARGS__ ")");