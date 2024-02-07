/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

interface INativePeer {
    void resetNative();

    EValue forward(EValue... inputs);

    EValue execute(String methodName, EValue... inputs);
}
