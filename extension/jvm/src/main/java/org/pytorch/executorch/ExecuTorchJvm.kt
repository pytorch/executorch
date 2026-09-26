/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/**
 * Entry point for using ExecuTorch on a desktop JVM (Linux, macOS, Windows).
 *
 * Call [init] once, before any other ExecuTorch API:
 * ```kotlin
 * ExecuTorchJvm.init()
 * val module = Module.load("model.pte")
 * ```
 *
 * [init] installs the desktop [JvmNativeLoaderDelegate], which extracts the packaged
 * `libexecutorch_jni` binary for the current OS/arch from the classpath (see [NativeLibraryLoader])
 * instead of relying on `java.library.path`. No reflection or ServiceLoader is involved — the
 * wiring is compile-checked.
 *
 * The corresponding per-platform native artifact must be on the classpath, e.g.
 * `org.pytorch:executorch-jvm:<version>:linux-x86_64`.
 */
object ExecuTorchJvm {
  @Volatile private var initialized = false

  /** Installs the desktop JVM native loading behavior. Idempotent. */
  @JvmStatic
  @Synchronized
  fun init() {
    if (initialized) {
      return
    }
    ExecuTorchRuntime.configureNativeLoading(JvmNativeLoaderDelegate(), "executorch_jni")
    initialized = true
  }
}
