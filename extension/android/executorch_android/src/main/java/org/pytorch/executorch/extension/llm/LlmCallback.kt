/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

import com.facebook.jni.annotations.DoNotStrip
import org.pytorch.executorch.annotations.Experimental

/**
 * Callback interface for Llm model. Users can implement this interface to receive the generated
 * tokens and statistics.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
interface LlmCallback {
  /**
   * Called when a new result is available from JNI. Users will keep getting onResult() invocations
   * until generate() finishes.
   *
   * @param result Last generated token
   */
  @DoNotStrip fun onResult(result: String)

  /**
   * Called when the statistics for the generate() is available.
   *
   * The result will be a JSON string. See extension/llm/stats.h for the field definitions.
   *
   * @param stats JSON string containing the statistics for the generate()
   */
  @DoNotStrip fun onStats(stats: String) {}

  /**
   * Called when an error occurs during generate().
   *
   * @param errorCode Error code from the ExecuTorch runtime (see
   *   [org.pytorch.executorch.ExecutorchRuntimeException])
   * @param message Human-readable error description
   */
  @DoNotStrip fun onError(errorCode: Int, message: String) {}
}
