/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

/** Immutable metadata for a method in a Module. */
public class MethodMetadata {
  private final String mName;
  private final String[] mBackends;

  MethodMetadata(String name, String[] backends) {
    mName = name;
    mBackends = backends;
  }

  /**
   * @return Method name
   */
  public String getName() {
    return mName;
  }

  /**
   * @return Backends used for this method
   */
  public String[] getBackends() {
    return mBackends;
  }
}
