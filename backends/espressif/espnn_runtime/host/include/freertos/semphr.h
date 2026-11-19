/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <freertos/FreeRTOS.h>

inline SemaphoreHandle_t xSemaphoreCreateMutexStatic(
    StaticSemaphore_t* storage) {
  return storage;
}

inline BaseType_t xSemaphoreTake(SemaphoreHandle_t handle, TickType_t) {
  handle->mutex.lock();
  return pdTRUE;
}

inline BaseType_t xSemaphoreGive(SemaphoreHandle_t handle) {
  handle->mutex.unlock();
  return pdTRUE;
}
