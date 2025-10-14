/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith

/** Unit tests for [ExecuTorchRuntime].  */
@RunWith(AndroidJUnit4::class)
class RuntimeInstrumentationTest {
    @Test
    fun testRuntimeApi() {
        val ops = ExecuTorchRuntime.getRegisteredOps()
        val backends = ExecuTorchRuntime.getRegisteredBackends()

        Assert.assertNotNull(ops)
        Assert.assertNotNull(backends)

        for (op in ops) {
            Assert.assertNotNull(op)
        }

        for (backend in backends) {
            Assert.assertNotNull(backend)
        }
    }
}
