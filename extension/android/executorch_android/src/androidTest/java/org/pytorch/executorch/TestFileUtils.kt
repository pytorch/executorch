package org.pytorch.executorch

import androidx.test.InstrumentationRegistry

/** Test File Utils */
object TestFileUtils {

    fun getTestFilePath(fileName: String): String {
        return InstrumentationRegistry.getInstrumentation()
            .targetContext
            .externalCacheDir
            .toString() + fileName
    }
}
