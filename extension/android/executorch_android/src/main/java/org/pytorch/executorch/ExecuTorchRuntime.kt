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
      loadSplitBackends()
    }

    /**
     * Loads the backends that were built as their own shared library.
     *
     * A split backend (EXECUTORCH_BUILD_XNNPACK_BACKEND_SHARED /
     * EXECUTORCH_BUILD_VULKAN_BACKEND_SHARED) links the runtime dynamically, so loading it here
     * pulls libexecutorch.so in through DT_NEEDED and its `register_backend` call resolves to the
     * one registry the runtime reads. Nothing links to the backend itself, so it has to be named
     * explicitly for it to be loaded at all.
     *
     * Each is absent in a build that linked the backend into libexecutorch.so, which is the
     * default, so a missing library is not an error.
     */
    private fun loadSplitBackends() {
      for (name in arrayOf("xnnpack_executorch_backend", "vulkan_executorch_backend")) {
        try {
          NativeLoader.loadLibrary(name)
        } catch (_: UnsatisfiedLinkError) {
          // Not part of this build.
        }
      }
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