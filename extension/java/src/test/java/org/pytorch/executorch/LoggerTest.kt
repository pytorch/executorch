/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/** Unit tests for the [Logger] / [Log] wiring shared by all platform modules. */
@RunWith(JUnit4::class)
class LoggerTest {

  private class RecordingLogger : Logger {
    val events = mutableListOf<Triple<String, String, String>>()

    override fun e(tag: String, msg: String) {
      events.add(Triple("e", tag, msg))
    }

    override fun w(tag: String, msg: String) {
      events.add(Triple("w", tag, msg))
    }

    override fun i(tag: String, msg: String) {
      events.add(Triple("i", tag, msg))
    }

    override fun d(tag: String, msg: String) {
      events.add(Triple("d", tag, msg))
    }
  }

  @Test
  fun installedLoggerReceivesAllLevels() {
    val recorder = RecordingLogger()
    Log.install(recorder)

    Log.e("tag", "error-message")
    Log.w("tag", "warning-message")
    Log.i("tag", "info-message")
    Log.d("tag", "debug-message")

    assertEquals(
        listOf(
            Triple("e", "tag", "error-message"),
            Triple("w", "tag", "warning-message"),
            Triple("i", "tag", "info-message"),
            Triple("d", "tag", "debug-message"),
        ),
        recorder.events,
    )
  }

  @Test
  fun consoleLoggerIsAValidDefault() {
    // The shared module must work without any platform module installing a logger.
    // ConsoleLogger writes to stdout/stderr and must not throw.
    val logger = ConsoleLogger()
    logger.e("tag", "message")
    logger.w("tag", "message")
    logger.i("tag", "message")
    logger.d("tag", "message")
  }
}
