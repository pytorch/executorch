/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import static org.junit.Assert.assertNotNull;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link ExecuTorchRuntime}. */
@RunWith(AndroidJUnit4.class)
public class RuntimeInstrumentationTest {

  @Test
  public void testRuntimeApi() {
    String[] ops = ExecuTorchRuntime.getRegisteredOps();
    String[] backends = ExecuTorchRuntime.getRegisteredBackends();

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
