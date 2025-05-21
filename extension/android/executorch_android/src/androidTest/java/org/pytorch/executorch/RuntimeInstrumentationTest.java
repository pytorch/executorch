/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertNotNull;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.runner.RunWith;
import org.junit.Test;

/** Unit tests for {@link Runtime}. */
@RunWith(AndroidJUnit4.class)
public class RuntimeInstrumentationTest {

    @Test
    public void testRuntimeApi() {
        assertTrue(Runtime.isInitialized());

        String[] ops = Runtime.getRegisteredOps();
        String[] backends = Runtime.getRegisteredBackends();

        assertNotNull(ops);
        assertNotNull(backends);

        for (String op : ops) {
            assertNotNull(op);
        }

        for (String backend : backends) {
            assertNotNull(backend);
        }
    }
}
