/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host stand-ins for the FreeRTOS calls the ESP-NN delegate makes: a mutex
// backed by std::mutex, and a no-op task yield.

#pragma once

#include <cstdint>
#include <mutex>

typedef uint32_t TickType_t;
typedef int BaseType_t;
#define portMAX_DELAY (static_cast<TickType_t>(0xffffffffU))
#define pdTRUE 1

struct StaticSemaphore_t {
  std::mutex mutex;
};
typedef StaticSemaphore_t* SemaphoreHandle_t;
