/* Copyright 2024-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>

void StartMeasurements();
void StopMeasurements(size_t num_inferences);
