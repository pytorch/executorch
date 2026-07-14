/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import java.lang.reflect.Method

/**
 * Platform-agnostic logging helper that forwards logs to android.util.Log when running on Android,
 * and falls back to standard error console logging when running on desktop JVMs.
 */
internal object Log {
  private var androidLogMethod: Method? = null
  private var lookupFailed = false

  fun e(tag: String, msg: String) {
    if (!lookupFailed && androidLogMethod == null) {
      try {
        val logClass = Class.forName("android.util.Log")
        androidLogMethod = logClass.getMethod("e", String::class.java, String::class.java)
      } catch (e: Exception) {
        lookupFailed = true
      }
    }

    val method = androidLogMethod
    if (method != null) {
      try {
        method.invoke(null, tag, msg)
        return
      } catch (e: Exception) {
        // Fallback to console printing if invocation fails
      }
    }

    System.err.println("[$tag] ERROR: $msg")
  }
}
