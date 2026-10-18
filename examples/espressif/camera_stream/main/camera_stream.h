/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace executorch {
namespace examples {
namespace espressif {

/**
 * Loads the embedded model, starts the camera and creates the inference
 * (core 1) and USB streaming (core 0) tasks, then returns. On a load or
 * camera failure it prints the reason and returns without starting tasks.
 */
void run_camera_stream();

} // namespace espressif
} // namespace examples
} // namespace executorch
