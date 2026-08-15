/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/**
 * Logging sink for the ExecuTorch Java APIs.
 *
 * Platform modules provide an implementation: the Android artifact installs an
 * `android.util.Log`-backed implementation at process start, while the desktop JVM artifact uses
 * [ConsoleLogger]. Implementations are installed via [Log.install] before any other ExecuTorch API
 * is used.
 */
interface Logger {
  /** Log an error message. */
  fun e(tag: String, msg: String)

  /** Log a warning message. */
  fun w(tag: String, msg: String)

  /** Log an info message. */
  fun i(tag: String, msg: String)

  /** Log a debug message. */
  fun d(tag: String, msg: String)
}

/**
 * Default [Logger] that writes warnings and errors to stderr and info/debug to stdout. Works on any
 * JVM and is the default sink unless a platform module installs another implementation.
 */
class ConsoleLogger : Logger {
  override fun e(tag: String, msg: String) {
    System.err.println("[$tag] ERROR: $msg")
  }

  override fun w(tag: String, msg: String) {
    System.err.println("[$tag] WARNING: $msg")
  }

  override fun i(tag: String, msg: String) {
    System.out.println("[$tag] INFO: $msg")
  }

  override fun d(tag: String, msg: String) {
    System.out.println("[$tag] DEBUG: $msg")
  }
}

/**
 * Internal logging facade used by the ExecuTorch Java API. Delegates to the installed [Logger].
 *
 * This is not part of the stable public API surface; applications should not call it directly.
 * Platform modules (Android, desktop JVM) call [install] once during initialization.
 */
object Log {
  @Volatile private var logger: Logger = ConsoleLogger()

  /** Installs the platform-specific [Logger]. Must be called before any logging occurs. */
  @JvmStatic
  fun install(custom: Logger) {
    logger = custom
  }

  @JvmStatic
  fun e(tag: String, msg: String) {
    logger.e(tag, msg)
  }

  @JvmStatic
  fun w(tag: String, msg: String) {
    logger.w(tag, msg)
  }

  @JvmStatic
  fun i(tag: String, msg: String) {
    logger.i(tag, msg)
  }

  @JvmStatic
  fun d(tag: String, msg: String) {
    logger.d(tag, msg)
  }
}
