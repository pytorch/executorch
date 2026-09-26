/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Utility for loading platform-specific native libraries on desktop JVMs.
 *
 * Native libraries are expected to be bundled in the classpath as:
 * ```
 * native/<os>/<arch>/<library-file>
 * ```
 *
 * For example:
 * - `native/linux/x86_64/libexecutorch_jni.so`
 * - `native/macos/aarch64/libexecutorch_jni.dylib`
 * - `native/windows/x86_64/executorch_jni.dll`
 *
 * The loader extracts the library to a temporary directory and loads it via [System.load]. The
 * temporary file is deleted on JVM exit.
 */
object NativeLibraryLoader {

  private val loaded = mutableSetOf<String>()

  /**
   * Loads the given native library for the current OS and architecture.
   *
   * @param libraryName the platform-independent library name (e.g. "executorch_jni")
   * @throws UnsatisfiedLinkError if the OS/arch is unsupported or the library is not found
   */
  @Synchronized
  fun load(libraryName: String) {
    if (libraryName in loaded) return

    val osName = System.getProperty("os.name")?.lowercase() ?: ""
    val arch = System.getProperty("os.arch")?.lowercase() ?: ""

    val osDir =
        when {
          "linux" in osName -> "linux"
          "mac" in osName || "darwin" in osName -> "macos"
          "win" in osName -> "windows"
          else -> throw UnsatisfiedLinkError("Unsupported OS: $osName")
        }

    val archDir =
        when {
          arch == "amd64" || arch == "x86_64" -> "x86_64"
          arch == "aarch64" || arch == "arm64" -> "aarch64"
          else -> throw UnsatisfiedLinkError("Unsupported architecture: $arch")
        }

    val fileName = System.mapLibraryName(libraryName)
    val resourcePath = "/native/$osDir/$archDir/$fileName"

    val inputStream =
        NativeLibraryLoader::class.java.getResourceAsStream(resourcePath)
            ?: throw UnsatisfiedLinkError(
                "Native library not found on classpath: $resourcePath. " +
                    "Add the appropriate platform-specific JAR " +
                    "(e.g. executorch-jvm-<version>-$osDir-$archDir.jar) to your dependencies."
            )

    try {
      val tempDir = File(System.getProperty("java.io.tmpdir"), "executorch-native")
      tempDir.mkdirs()
      val tempFile = File(tempDir, fileName)
      tempFile.deleteOnExit()

      FileOutputStream(tempFile).use { output -> inputStream.use { input -> input.copyTo(output) } }

      System.load(tempFile.absolutePath)
      loaded.add(libraryName)
    } catch (e: IOException) {
      throw UnsatisfiedLinkError("Failed to extract native library: ${e.message}")
    }
  }
}
