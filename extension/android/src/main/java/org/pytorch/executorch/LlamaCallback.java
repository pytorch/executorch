/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.annotations.DoNotStrip;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Callback interface for Llama model. Users can implement this interface to receive the generated
 * tokens and statistics.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public interface LlamaCallback {
  /**
   * Called when a new result is available from JNI. Users will keep getting onResult() invocations
   * until generate() finishes.
   *
   * @param result Last generated token
   */
  @DoNotStrip
  public void onResult(String result);

  /**
   * Called when the statistics for the generate() is available.
   *
   * @param tps Tokens/second for generated tokens.
   */
  @DoNotStrip
  public void onStats(float tps);
}
