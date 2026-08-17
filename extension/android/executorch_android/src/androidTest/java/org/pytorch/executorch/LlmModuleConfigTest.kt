/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.extension.llm.LlmModuleConfig

/** Tests for [LlmModuleConfig]. */
@RunWith(AndroidJUnit4::class)
class LlmModuleConfigTest {

  @Test
  fun testDataPathDefaultsToNull() {
    // An empty default reaches the runner as a real path, which then fails to open and takes the
    // whole load down. Absent has to be null.
    val config =
        LlmModuleConfig.create().modulePath("/model.pte").tokenizerPath("/tokenizer.json").build()
    assertNull(config.dataPath)
  }

  @Test
  fun testDataPathRoundTrips() {
    val config =
        LlmModuleConfig.create()
            .modulePath("/model.pte")
            .tokenizerPath("/tokenizer.json")
            .dataPath("/weights.ptd")
            .build()
    assertEquals("/weights.ptd", config.dataPath)
  }

  @Test
  fun testDefaults() {
    val config =
        LlmModuleConfig.create().modulePath("/model.pte").tokenizerPath("/tokenizer.json").build()
    assertEquals("/model.pte", config.modulePath)
    assertEquals("/tokenizer.json", config.tokenizerPath)
    assertEquals(LlmModuleConfig.MODEL_TYPE_TEXT, config.modelType)
    assertEquals(LlmModuleConfig.LOAD_MODE_MMAP, config.loadMode)
  }
}
