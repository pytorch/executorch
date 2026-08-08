/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/** Desktop JVM logging helper that prints warnings and errors to standard error/out. */
internal object Log {
  fun e(tag: String, msg: String) {
    System.err.println("[$tag] ERROR: $msg")
  }

  fun w(tag: String, msg: String) {
    System.err.println("[$tag] WARNING: $msg")
  }

  fun i(tag: String, msg: String) {
    System.out.println("[$tag] INFO: $msg")
  }

  fun d(tag: String, msg: String) {
    System.out.println("[$tag] DEBUG: $msg")
  }
}
