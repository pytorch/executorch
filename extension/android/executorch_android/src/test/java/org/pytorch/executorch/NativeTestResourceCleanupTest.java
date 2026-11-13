/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

public class NativeTestResourceCleanupTest {
  @Test
  public void normalCloseDoesNotAbort() {
    AtomicInteger closes = new AtomicInteger();
    AtomicInteger aborts = new AtomicInteger();
    NativeTestResourceCleanup.closeOrAbort(closes::incrementAndGet, aborts::incrementAndGet);
    assertEquals(1, closes.get());
    assertEquals(0, aborts.get());
  }

  @Test
  public void busyNativeMethodAbortsAndCannotBecomePassIfAbortReturns() {
    IllegalStateException busy =
        new IllegalStateException("Cannot close while method is executing");
    AtomicInteger aborts = new AtomicInteger();
    IllegalStateException failure =
        assertThrows(
            IllegalStateException.class,
            () ->
                NativeTestResourceCleanup.closeOrAbort(
                    () -> {
                      throw busy;
                    },
                    aborts::incrementAndGet));
    assertSame(busy, failure);
    assertEquals(1, aborts.get());
  }

  @Test
  public void abortFailureDoesNotHideUnsafeClose() {
    IllegalStateException busy = new IllegalStateException("Native method is executing");
    RuntimeException abortFailure = new RuntimeException("Could not abort");
    IllegalStateException failure =
        assertThrows(
            IllegalStateException.class,
            () ->
                NativeTestResourceCleanup.closeOrAbort(
                    () -> {
                      throw busy;
                    },
                    () -> {
                      throw abortFailure;
                    }));
    assertSame(busy, failure);
    assertEquals(1, failure.getSuppressed().length);
    assertSame(abortFailure, failure.getSuppressed()[0]);
  }

  @Test
  public void otherFailuresRemainFailuresWithoutInvokingAbort() {
    IllegalArgumentException original = new IllegalArgumentException("Other cleanup failure");
    AtomicInteger aborts = new AtomicInteger();
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NativeTestResourceCleanup.closeOrAbort(
                    () -> {
                      throw original;
                    },
                    aborts::incrementAndGet));
    assertSame(original, failure);
    assertEquals(0, aborts.get());
  }
}
