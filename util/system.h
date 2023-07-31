/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * OS-specific #defines that are helpful for writing portable utility targets,
 * but should not be part of the core Executorch API.
 */

#pragma once

// Defined when the target system supports mmap().
#ifndef ET_MMAP_SUPPORTED
#if defined(__linux__) || defined(__APPLE__)
#define ET_MMAP_SUPPORTED 1
#endif // __linux__ || __APPLE__
#endif // !ET_MMAP_SUPPORTED
