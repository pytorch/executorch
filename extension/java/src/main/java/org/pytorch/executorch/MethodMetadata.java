/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

/** Helper class to access the metadata for a method from a Module */
public class MethodMetadata {
  private String mName;

  private String[] mBackends;

  MethodMetadata setName(String name) {
    mName = name;
    return this;
  }

  /**
   * @return Method name
   */
  public String getName() {
    return mName;
  }

  MethodMetadata setBackends(String[] backends) {
    mBackends = backends;
    return this;
  }

  /**
   * @return Backends used for this method
   */
  public String[] getBackends() {
    return mBackends;
  }
}
