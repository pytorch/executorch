package org.pytorch.executorch

import androidx.test.InstrumentationRegistry
import java.io.File
import java.io.FileNotFoundException
import org.apache.commons.io.FileUtils

/** Test File Utils */
object TestFileUtils {

  private const val PUSH_DIR = "/data/local/tmp/executorch"

  fun getTestFilePath(fileName: String): String {
    return InstrumentationRegistry.getInstrumentation().targetContext.cacheDir.toString() + fileName
  }

  /**
   * Resolve a test file by checking the adb push directory first, then falling back to APK
   * resources. Returns an absolute file path usable by native code.
   */
  fun prepareTestFile(caller: Class<*>, resourceName: String): String {
    val pushed = File("$PUSH_DIR$resourceName")
    if (pushed.exists()) {
      return pushed.absolutePath
    }
    val cached = File(getTestFilePath(resourceName))
    if (!cached.exists()) {
      val stream =
          caller.getResourceAsStream(resourceName)
              ?: throw FileNotFoundException(
                  "$resourceName not found in $PUSH_DIR or APK resources"
              )
      FileUtils.copyInputStreamToFile(stream, cached)
      stream.close()
    }
    return cached.absolutePath
  }
}
