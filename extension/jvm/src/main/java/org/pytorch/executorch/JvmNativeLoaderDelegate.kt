/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.soloader.nativeloader.NativeLoaderDelegate

/**
 * Desktop JVM-specific [NativeLoaderDelegate] that delegates library loading to
 * [NativeLibraryLoader].
 *
 * This implementation maps requests to load "executorch" to the actual JNI library "executorch_jni"
 * built for desktop platforms.
 */
class JvmNativeLoaderDelegate : NativeLoaderDelegate {
  override fun loadLibrary(shortName: String, flags: Int): Boolean {
    val libraryToLoad = if (shortName == "executorch") "executorch_jni" else shortName
    NativeLibraryLoader.load(libraryToLoad)
    return true
  }

  override fun getLibraryPath(libName: String): String? {
    return null
  }

  override fun getSoSourcesVersion(): Int {
    return 0
  }
}
