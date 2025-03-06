/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// This header is a stub left behind after the move to
// executorch/runtime/kernel. Depend on this target and include this
// header if you have a hard requirement for threading; if you want to
// cleanly use parallelization if available, then depend on and use
// the below header instead.
#include <executorch/runtime/kernel/thread_parallel_interface.h>
