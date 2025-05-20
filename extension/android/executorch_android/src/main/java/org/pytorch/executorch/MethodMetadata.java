/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

/**
 * Helper class to access the metadata for a method from a Module
 */
public class MethodMetadata {
    private String name;

    public String getName() {
        return name;
    }
}
