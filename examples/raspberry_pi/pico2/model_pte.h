/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t model_pte[] __attribute__((aligned(8)));
extern const unsigned int model_pte_len;

#ifdef __cplusplus
}
#endif
