/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_WRAPPER
#ifdef USE_VULKAN_VOLK
#ifdef VK_ANDROID_external_memory_android_hardware_buffer
#include <android/hardware_buffer.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

#include <volk.h>
#else
#include <vulkan_wrapper.h>
#endif /* USE_VULKAN_VOLK */
#else
#include <vulkan/vulkan.h>
#endif /* USE_VULKAN_WRAPPER */
