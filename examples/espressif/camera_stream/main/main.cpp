/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <camera_stream.h>

namespace {

void camera_task(void*) {
  executorch::examples::espressif::run_camera_stream();
  vTaskDelete(nullptr);
}

} // namespace

extern "C" void app_main() {
  xTaskCreatePinnedToCore(
      camera_task, "camera_setup", 16 * 1024, nullptr, 5, nullptr, 0);
}
