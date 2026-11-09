/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Performance monitoring helpers for Espressif ESP32/ESP32-S3.
 *
 * Uses ESP-IDF's 64-bit monotonic timer for elapsed wall time.
 */

void StartMeasurements();
void StopMeasurements(int num_inferences);
