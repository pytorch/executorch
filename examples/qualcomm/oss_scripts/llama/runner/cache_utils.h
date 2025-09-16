/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

enum CacheMode {
  StaticCahce = 0,
  // For models with global/local attention architecture (e.g., Gemma3),
  HybridCache,
};
