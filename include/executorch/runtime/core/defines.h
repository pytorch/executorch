/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Contains preprocessor definitions used by ExecuTorch core.
 */

#pragma once

// Enable ET_ENABLE_ENUM_STRINGS by default. This option gates inclusion of
// enum string names and can be disabled by explicitly setting it to 0.
#ifndef ET_ENABLE_ENUM_STRINGS
#define ET_ENABLE_ENUM_STRINGS 1
#endif
