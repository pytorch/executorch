/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Example ESP-IDF main component.
 *
 * The app_main() defined below performs optional initialization and then
 * calls executor_runner_main().
 *
 * If you want to customize the runner behavior, you can modify the
 * app_main() implementation here (e.g., add initialization or cleanup)
 * while still delegating to executor_runner_main().
 */


#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"

extern void executor_runner_main(void);

extern "C" void app_main(void) {
    printf("Starting executorch runner !\n");
    fflush(stdout);
    // Custom initialization here
    executor_runner_main();
    for (int i = 5; i >= 0; i--) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    esp_restart();
}
