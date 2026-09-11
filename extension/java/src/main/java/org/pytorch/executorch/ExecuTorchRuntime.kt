/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.jni.annotations.DoNotStrip
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.File

/** Class for entire ExecuTorch Runtime related functions. */
class ExecuTorchRuntime private constructor() {

  companion object {
    init {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(SystemDelegate())
      }
      // Loads libexecutorch.so from jniLibs
      NativeLoader.loadLibrary("executorch")
    }

    private val sInstance = ExecuTorchRuntime()

    /** Get the runtime instance. */
    @JvmStatic fun getRuntime(): ExecuTorchRuntime = sInstance

    /**
     * Validates that the given path points to a readable file.
     *
     * @throws IllegalArgumentException if the path is null, does not exist, is not a file, or is
     *   not readable.
     */
    @JvmStatic
    fun validateFilePath(path: String?, description: String) {
      if (path == null) {
        throw IllegalArgumentException("Cannot load $description: path is null")
      }
      val file = File(path)
      if (!file.exists()) {
        throw IllegalArgumentException("Cannot load $description: path does not exist: $path")
      }
      if (!file.isFile) {
        throw IllegalArgumentException("Cannot load $description: path is not a file: $path")
      }
      if (!file.canRead()) {
        throw IllegalArgumentException("Cannot load $description: path is not readable: $path")
      }
    }

    /** Get all registered ops. */
    @DoNotStrip @JvmStatic external fun getRegisteredOps(): Array<String>

    /** Get all registered backends. */
    @DoNotStrip @JvmStatic external fun getRegisteredBackends(): Array<String>
  }
}
