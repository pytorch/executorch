/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/**
 * Desktop JVM [Logger] implementation that writes to standard output/error streams.
 *
 * Discovered automatically via [java.util.ServiceLoader] when the executorch-jvm JAR is on the
 * classpath.
 */
class ConsoleLogger : Logger {
  override fun e(tag: String, msg: String) = System.err.println("[$tag] ERROR: $msg")

  override fun w(tag: String, msg: String) = System.err.println("[$tag] WARN:  $msg")

  override fun i(tag: String, msg: String) = System.out.println("[$tag] INFO:  $msg")

  override fun d(tag: String, msg: String) = System.out.println("[$tag] DEBUG: $msg")
}
