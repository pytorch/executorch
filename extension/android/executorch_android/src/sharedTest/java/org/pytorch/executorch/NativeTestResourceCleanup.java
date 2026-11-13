/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch;

/** Prevents the next test from reusing native state when a timed-out method is still active. */
public final class NativeTestResourceCleanup {
  private NativeTestResourceCleanup() {}

  public static void closeOrAbort(Runnable close, Runnable abortProcess) {
    try {
      close.run();
    } catch (IllegalStateException unsafeClose) {
      try {
        abortProcess.run();
      } catch (RuntimeException abortFailure) {
        unsafeClose.addSuppressed(abortFailure);
      }
      throw unsafeClose;
    }
  }
}
