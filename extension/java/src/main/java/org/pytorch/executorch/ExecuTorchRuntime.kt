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
import com.facebook.soloader.nativeloader.NativeLoaderDelegate
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.File

/** Class for entire ExecuTorch Runtime related functions. */
class ExecuTorchRuntime private constructor() {

  companion object {
    private val initLock = Any()

    // Defaults preserve the long-standing Android behavior: soloader's SystemDelegate and the
    // "executorch" library name. Platform modules (e.g. the desktop JVM artifact) override these
    // via configureNativeLoading() before any ExecuTorch API is touched.
    @Volatile private var nativeLoaderDelegate: NativeLoaderDelegate = SystemDelegate()
    @Volatile private var nativeLibraryName: String = "executorch"

    init {
      ensureNativeLibraryLoaded()
    }

    private val sInstance = ExecuTorchRuntime()

    /** Get the runtime instance. */
    @JvmStatic fun getRuntime(): ExecuTorchRuntime = sInstance

    /**
     * Configures how the ExecuTorch native library is loaded. Intended for platform modules (e.g.
     * the desktop JVM artifact, which extracts a packaged .so/.dylib/.dll from the classpath) —
     * Android applications do not need to call this.
     *
     * Must be called before any other ExecuTorch API, i.e. before the native library has been
     * loaded.
     *
     * @param delegate the soloader [NativeLoaderDelegate] used to resolve and load the library.
     * @param libraryName the platform-independent library name passed to the delegate.
     */
    @JvmStatic
    fun configureNativeLoading(delegate: NativeLoaderDelegate, libraryName: String) {
      synchronized(initLock) {
        check(!NativeLoader.isInitialized()) {
          "configureNativeLoading must be called before the ExecuTorch native library is loaded"
        }
        nativeLoaderDelegate = delegate
        nativeLibraryName = libraryName
      }
    }

    /** Initializes soloader (once) and loads the ExecuTorch native library. Idempotent. */
    @JvmStatic
    fun ensureNativeLibraryLoaded() {
      synchronized(initLock) {
        if (!NativeLoader.isInitialized()) {
          NativeLoader.init(nativeLoaderDelegate)
        }
        // Loads libexecutorch.so from jniLibs on Android; delegated to the platform module's
        // NativeLoaderDelegate elsewhere.
        NativeLoader.loadLibrary(nativeLibraryName)
      }
    }

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
