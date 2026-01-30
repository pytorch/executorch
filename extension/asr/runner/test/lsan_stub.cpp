/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// lsan_stub.cpp - Fix for macOS LSan linking issue
#if defined(__APPLE__) && defined(__arm64__)
extern "C" {
// Provide stub for LSan symbol that macOS doesn't implement
int __lsan_is_turned_off() {
  return 1;
}
}
#endif
