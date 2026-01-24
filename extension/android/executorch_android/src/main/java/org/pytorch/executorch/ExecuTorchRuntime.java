/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;



/** Class for entire ExecuTorch Runtime related functions. */
public class ExecuTorchRuntime {

  static {
    // Loads libexecutorch.so from jniLibs
    System.loadLibrary("executorch");
  }

  private static final ExecuTorchRuntime sInstance = new ExecuTorchRuntime();

  private ExecuTorchRuntime() {}

  /** Get the runtime instance. */
  public static ExecuTorchRuntime getRuntime() {
    return sInstance;
  }

  /** Get all registered ops. */

  public static native String[] getRegisteredOps();

  /** Get all registered backends. */

  public static native String[] getRegisteredBackends();
}
