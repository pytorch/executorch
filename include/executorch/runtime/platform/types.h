/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Public types used by the ExecuTorch Platform Abstraction Layer.
 */

#pragma once

// Use C-style includes so that C code can include this header.
#include <stdint.h>

extern "C" {

/// Platform timestamp in system ticks.
typedef uint64_t et_timestamp_t;

} // extern "C"
