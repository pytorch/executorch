/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//
// Do NOT include vk_mem_alloc.h directly.
// Always include this file (vma_api.h) instead.
//

#ifdef ETVK_USE_META_VMA

// Match the config from the shared third-party
// VulkanMemoryAllocatorInstantiated library so that struct layouts agree.
// We do NOT define VMA_IMPLEMENTATION here; the third-party static lib
// provides it.
#undef VMA_STATIC_VULKAN_FUNCTIONS
#undef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1002000

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4101 4189)
#endif

#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#else // !ETVK_USE_META_VMA

#define VMA_VULKAN_VERSION 1000000

#ifdef USE_VULKAN_WRAPPER
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#else
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif /* USE_VULKAN_WRAPPER */

#define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (4ull * 1024 * 1024)
#define VMA_SMALL_HEAP_MAX_SIZE (256ull * 1024 * 1024)

#define VMA_STATS_STRING_ENABLED 0

#ifdef VULKAN_DEBUG
#define VMA_DEBUG_ALIGNMENT 4096
#define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY 0
#define VMA_DEBUG_DETECT_CORRUPTION 1
#define VMA_DEBUG_GLOBAL_MUTEX 1
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#define VMA_DEBUG_MARGIN 64
#define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY 256
#define VMA_RECORDING_ENABLED 1

#define VMA_DEBUG_LOG(format, ...)
/*
#define VMA_DEBUG_LOG(format, ...) do { \
    printf(format, __VA_ARGS__); \
    printf("\n"); \
} while(false)
*/
#endif /* VULKAN_DEBUG */

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4101 4189)
#endif

#include <include/vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif // ETVK_USE_META_VMA
