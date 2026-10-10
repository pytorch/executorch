/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import android.util.Log
import com.facebook.jni.annotations.DoNotStrip
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import dalvik.system.BaseDexClassLoader
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

    private const val TAG = "ExecuTorchRuntime"

    private val SPLIT_BACKEND_LIBRARIES =
        arrayOf("xnnpack_executorch_backend", "vulkan_executorch_backend")

    /**
     * Loads the backends that were built as their own shared library.
     *
     * A split backend (EXECUTORCH_BUILD_XNNPACK_BACKEND_SHARED /
     * EXECUTORCH_BUILD_VULKAN_BACKEND_SHARED) links the runtime dynamically, so loading it here
     * pulls libexecutorch.so in through DT_NEEDED and its `register_backend` call resolves to the
     * one registry the runtime reads. Nothing links to the backend itself, so it has to be named
     * explicitly for it to be loaded at all.
     *
     * A library that is not packaged is the normal case and not an error: the default build links
     * the backend into libexecutorch.so instead. A library that is packaged but fails to load is a
     * different thing entirely, so the two are distinguished rather than both being swallowed.
     */
    private fun loadSplitBackends() {
      for (name in SPLIT_BACKEND_LIBRARIES) {
        try {
          NativeLoader.loadLibrary(name)
        } catch (e: UnsatisfiedLinkError) {
          val path = packagedLibraryPath(name)
          if (path == null) {
            Log.d(
                TAG,
                "No lib$name.so in this build; the backend it provides is either linked into " +
                    "libexecutorch.so or not included",
            )
          } else {
            // Packaged but unloadable, typically an unresolved symbol or a
            // dependency mismatch. Swallowing this would leave the delegate
            // unavailable at run time with nothing to point at.
            Log.e(TAG, "Failed to load $path; the backend it provides will be unavailable", e)
          }
        }
      }
    }

    /**
     * The path of lib[name].so in this APK, or null when it is not packaged.
     *
     * Returns null for a class loader that cannot be asked, which no Android application has, so an
     * unexpected loader reports "not packaged" rather than failing the load.
     */
    private fun packagedLibraryPath(name: String): String? =
        (ExecuTorchRuntime::class.java.classLoader as? BaseDexClassLoader)?.findLibrary(name)

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
