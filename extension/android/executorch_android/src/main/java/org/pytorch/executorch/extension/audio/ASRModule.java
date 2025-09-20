/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.audio;
import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import java.io.File;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.annotations.Experimental;

/**
 * ASRModule is a wrapper around the Executorch ASR runners like Whisper runner.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class ASRModule {

  @DoNotStrip private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(
      String modulePath, String tokenizerPath);

  public ASRModule(
      String modulePath, String tokenizerPath) {
    ExecuTorchRuntime runtime = ExecuTorchRuntime.getRuntime();

    File modelFile = new File(modulePath);
    if (!modelFile.canRead() || !modelFile.isFile()) {
      throw new RuntimeException("Cannot load model path " + modulePath);
    }
    File tokenizerFile = new File(tokenizerPath);
    if (!tokenizerFile.canRead() || !tokenizerFile.isFile()) {
      throw new RuntimeException("Cannot load tokenizer path " + tokenizerPath);
    }
    mHybridData = initHybrid(modulePath, tokenizerPath);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  @DoNotStrip
  public native int transcribe(
      int seqLen,
      byte[][] inputs,
      LlmCallback callback,
      int n_bins,
      int n_frames);

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
