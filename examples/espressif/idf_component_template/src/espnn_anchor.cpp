/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <espnn/espnn_backend.h>

// __executorch_finalize adds -u et_anchor_backend_espnn when a model uses the
// ESP-NN delegate. The reference pulls in espnn_backend.cpp, whose static
// initializer registers the backend.
extern "C" bool (*const et_anchor_backend_espnn)() =
    &executorch::backends::espressif::register_espnn_backend;
