/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/**
 * Platform logging facade used by the shared sources under extension/java.
 *
 * <p>The shared sources reference [Log] by simple name (same package, no import), and each platform
 * artifact compiles its own implementation:
 * <ul>
 * <li>Android (this file): delegates to [android.util.Log], so logcat output — tags, priorities,
 *   and messages — is byte-for-byte identical to logging directly from the caller.
 * <li>Desktop JVM (extension/jvm, added separately): a console-backed implementation.
 * </ul>
 *
 * The shared source set itself contains no [Log] definition, so there is exactly one implementation
 * per artifact and no duplicate-class collision.
 */
internal object Log {
  @JvmStatic fun v(tag: String, msg: String): Int = android.util.Log.v(tag, msg)

  @JvmStatic fun d(tag: String, msg: String): Int = android.util.Log.d(tag, msg)

  @JvmStatic fun i(tag: String, msg: String): Int = android.util.Log.i(tag, msg)

  @JvmStatic fun w(tag: String, msg: String): Int = android.util.Log.w(tag, msg)

  @JvmStatic fun w(tag: String, msg: String, tr: Throwable): Int = android.util.Log.w(tag, msg, tr)

  @JvmStatic fun e(tag: String, msg: String): Int = android.util.Log.e(tag, msg)

  @JvmStatic fun e(tag: String, msg: String, tr: Throwable): Int = android.util.Log.e(tag, msg, tr)
}
