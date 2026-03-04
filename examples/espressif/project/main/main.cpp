/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Example ESP-IDF main component.
 *
 * The actual app_main() is defined in esp_executor_runner.cpp.
 * This file can be used to add project-specific initialization
 * or to override the default behavior.
 *
 * If you want to customize the runner, you can:
 * 1. Remove app_main() from esp_executor_runner.cpp (remove ESP_PLATFORM guard)
 * 2. Define your own app_main() here that calls executor_runner_main()
 */

// The app_main() entry point is provided by esp_executor_runner.cpp
// when built with ESP_PLATFORM defined.
//
// Alternatively, uncomment below to define your own entry point:
//
// extern void executor_runner_main(void);
//
// extern "C" void app_main(void) {
//     // Custom initialization here
//     executor_runner_main();
// }
