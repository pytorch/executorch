/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Empty translation unit so SHARED libraries that only need to whole-archive
// other static archives (e.g. libvulkan_executorch_backend.so) have at least
// one source file for CMake to invoke the linker on.
