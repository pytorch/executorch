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
 * Desktop JVM [NativeLoaderDelegate] that loads native libraries via [NativeLibraryLoader], which
 * extracts the binary for the current OS/arch from the classpath.
 *
 * Installed by [ExecuTorchJvm.init]; the shared runtime passes the configured library name
 * ("executorch_jni") straight through.
 */
class JvmNativeLoaderDelegate : NativeLoaderDelegate {
  override fun loadLibrary(shortName: String, flags: Int): Boolean {
    NativeLibraryLoader.load(shortName)
    return true
  }

  override fun getLibraryPath(libName: String): String? {
    return null
  }

  override fun getSoSourcesVersion(): Int {
    return 0
  }
}
