/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

/** Interface for the native peer object for entry points to the Module */
interface INativePeer {
  /** Clean up the native resources associated with this instance */
  void resetNative();

  /** Run a "forward" call with the given inputs */
  EValue[] forward(EValue... inputs);

  /** Run an arbitrary method on the module */
  EValue[] execute(String methodName, EValue... inputs);

  /**
   * Load a method on this module.
   *
   * @return the Error code if there was an error loading the method
   */
  int loadMethod(String methodName);
}
