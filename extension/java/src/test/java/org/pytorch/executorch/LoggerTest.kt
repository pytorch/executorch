/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import org.junit.Test

/** Basic sanity test for [Log] facade. */
class LoggerTest {

  @Test
  fun testFallbackLogging() {
    // Under :executorch_java tests, there is no platform Logger registered
    // in META-INF/services, so it should fall back to Console fallback.
    Log.d("LoggerTest", "Test debug log")
    Log.i("LoggerTest", "Test info log")
    Log.w("LoggerTest", "Test warn log")
    Log.e("LoggerTest", "Test error log")
  }
}
