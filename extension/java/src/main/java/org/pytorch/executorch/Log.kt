/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import java.util.ServiceLoader

/**
 * Platform-agnostic logging interface.
 *
 * Platform-specific modules (Android, JVM desktop) provide implementations via [ServiceLoader] SPI.
 * If no implementation is found on the classpath, a built-in [FallbackLogger] that writes to
 * stderr/stdout is used.
 */
interface Logger {
  fun e(tag: String, msg: String)

  fun w(tag: String, msg: String)

  fun i(tag: String, msg: String)

  fun d(tag: String, msg: String)
}

/**
 * Internal logging facade used throughout the ExecuTorch Java API.
 *
 * Delegates to a [Logger] implementation discovered via [ServiceLoader], falling back to a console
 * logger if none is available.
 */
internal object Log {
  private val delegate: Logger by lazy {
    ServiceLoader.load(Logger::class.java).firstOrNull() ?: FallbackLogger()
  }

  fun e(tag: String, msg: String) = delegate.e(tag, msg)

  fun w(tag: String, msg: String) = delegate.w(tag, msg)

  fun i(tag: String, msg: String) = delegate.i(tag, msg)

  fun d(tag: String, msg: String) = delegate.d(tag, msg)
}

/** Default fallback if no platform-specific [Logger] is on the classpath. */
private class FallbackLogger : Logger {
  override fun e(tag: String, msg: String) = System.err.println("[$tag] ERROR: $msg")

  override fun w(tag: String, msg: String) = System.err.println("[$tag] WARN:  $msg")

  override fun i(tag: String, msg: String) = System.out.println("[$tag] INFO:  $msg")

  override fun d(tag: String, msg: String) = System.out.println("[$tag] DEBUG: $msg")
}
