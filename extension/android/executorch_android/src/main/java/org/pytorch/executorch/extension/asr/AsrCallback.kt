/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.asr

import org.pytorch.executorch.annotations.Experimental

/**
 * Callback interface for ASR (Automatic Speech Recognition) module. Users can implement this
 * interface to receive the transcribed tokens as they are generated.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
interface AsrCallback {
  /**
   * Called when a new token is available from JNI. Users will keep getting onToken() invocations
   * until transcription finishes (when the transcribe method returns).
   *
   * @param token The decoded text token
   */
  fun onToken(token: String)
}
