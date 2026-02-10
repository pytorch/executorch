/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Utility class to load native libraries for ExecuTorch on desktop platforms (Linux, macOS,
 * Windows).
 *
 * <p>This class handles loading native libraries either from the system library path or by
 * extracting them from the JAR resources to a temporary directory.
 */
public final class NativeLibraryLoader {

  private static final Set<String> loadedLibraries = new HashSet<>();
  private static File tempDir = null;
  private static boolean initialized = false;

  private NativeLibraryLoader() {}

  /**
   * Load a native library by name.
   *
   * <p>First attempts to load from the system library path using {@link System#loadLibrary}. If
   * that fails, attempts to extract the library from JAR resources and load it.
   *
   * @param libraryName the library name without platform-specific prefix/suffix (e.g.,
   *     "executorch_jni" not "libexecutorch_jni.so")
   */
  public static synchronized void loadLibrary(String libraryName) {
    if (loadedLibraries.contains(libraryName)) {
      return;
    }

    // First, try to load from system library path
    try {
      System.loadLibrary(libraryName);
      loadedLibraries.add(libraryName);
      return;
    } catch (UnsatisfiedLinkError e) {
      // Fall through to try loading from JAR
    }

    // Try to load from JAR resources
    String platformLibName = getPlatformLibraryName(libraryName);
    String resourcePath = "/native/" + getOsArch() + "/" + platformLibName;

    try {
      File libFile = extractLibraryFromResources(resourcePath, platformLibName);
      if (libFile != null) {
        System.load(libFile.getAbsolutePath());
        loadedLibraries.add(libraryName);
        return;
      }
    } catch (IOException e) {
      // Fall through to final error
    }

    // Last resort: try system load again to get a useful error message
    System.loadLibrary(libraryName);
    loadedLibraries.add(libraryName);
  }

  /**
   * Get the platform-specific library file name.
   *
   * @param libraryName the base library name
   * @return the platform-specific file name (e.g., "libfoo.so" on Linux)
   */
  private static String getPlatformLibraryName(String libraryName) {
    String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);

    if (osName.contains("mac") || osName.contains("darwin")) {
      return "lib" + libraryName + ".dylib";
    } else if (osName.contains("win")) {
      return libraryName + ".dll";
    } else {
      // Default to Linux/Unix style
      return "lib" + libraryName + ".so";
    }
  }

  /**
   * Get the OS and architecture string for resource paths.
   *
   * @return a string like "linux-x86_64", "darwin-aarch64", or "windows-x86_64"
   */
  private static String getOsArch() {
    String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);

    String os;
    if (osName.contains("mac") || osName.contains("darwin")) {
      os = "darwin";
    } else if (osName.contains("win")) {
      os = "windows";
    } else {
      os = "linux";
    }

    // Normalize architecture names
    if (arch.equals("amd64") || arch.equals("x86_64")) {
      arch = "x86_64";
    } else if (arch.equals("aarch64") || arch.equals("arm64")) {
      arch = "aarch64";
    }

    return os + "-" + arch;
  }

  /**
   * Extract a library from JAR resources to a temporary file.
   *
   * @param resourcePath the path within the JAR
   * @param fileName the file name to use in the temp directory
   * @return the extracted File, or null if resource not found
   * @throws IOException if extraction fails
   */
  private static File extractLibraryFromResources(String resourcePath, String fileName)
      throws IOException {
    InputStream in = NativeLibraryLoader.class.getResourceAsStream(resourcePath);
    if (in == null) {
      return null;
    }

    try {
      if (tempDir == null) {
        tempDir = createTempDirectory();
      }

      File outFile = new File(tempDir, fileName);
      try (OutputStream out = new FileOutputStream(outFile)) {
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
          out.write(buffer, 0, bytesRead);
        }
      }

      // Make the library executable on Unix-like systems
      outFile.setExecutable(true);

      // Register for cleanup on JVM exit
      outFile.deleteOnExit();

      return outFile;
    } finally {
      in.close();
    }
  }

  /**
   * Create a temporary directory for extracted native libraries.
   *
   * @return the temporary directory
   * @throws IOException if creation fails
   */
  private static File createTempDirectory() throws IOException {
    File temp = File.createTempFile("executorch_native", "");
    if (!temp.delete()) {
      throw new IOException("Failed to delete temp file: " + temp);
    }
    if (!temp.mkdir()) {
      throw new IOException("Failed to create temp directory: " + temp);
    }
    temp.deleteOnExit();
    return temp;
  }
}
