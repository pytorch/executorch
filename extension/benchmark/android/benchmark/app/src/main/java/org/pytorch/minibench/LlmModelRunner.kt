/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Message
import android.util.Log
import org.pytorch.executorch.ExecutorchRuntimeException
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

/** A helper class to handle all model running logic within this class. */
class LlmModelRunner(
    modelFilePath: String,
    tokenizerFilePath: String,
    temperature: Float,
    val callback: LlmModelRunnerCallback,
) : LlmCallback {

  val module: LlmModule = LlmModule(modelFilePath, tokenizerFilePath, temperature)
  private val handlerThread: HandlerThread = HandlerThread("LlmModelRunner")
  private val handler: Handler

  init {
    handlerThread.start()
    handler = LlmModelRunnerHandler(handlerThread.looper, this)
    handler.sendEmptyMessage(LlmModelRunnerHandler.MESSAGE_LOAD_MODEL)
  }

  fun generate(prompt: String): Int {
    val msg = Message.obtain(handler, LlmModelRunnerHandler.MESSAGE_GENERATE, prompt)
    msg.sendToTarget()
    return 0
  }

  fun stop() {
    module.stop()
  }

  override fun onResult(result: String) {
    callback.onTokenGenerated(result)
  }

  override fun onStats(stats: String) {
    callback.onStats(stats)
  }
}

private class LlmModelRunnerHandler(
    looper: Looper,
    private val runner: LlmModelRunner,
) : Handler(looper) {

  override fun handleMessage(msg: Message) {
    when (msg.what) {
      MESSAGE_LOAD_MODEL -> {
        val status =
            try {
              runner.module.load()
              0
            } catch (e: ExecutorchRuntimeException) {
              e.errorCode
            } catch (e: Exception) {
              -1
            }
        runner.callback.onModelLoaded(status)
      }
      MESSAGE_GENERATE -> {
        try {
          runner.module.generate(msg.obj as String, runner)
        } catch (e: Exception) {
          Log.e("LlmModelRunner", "generate() failed", e)
        }
        runner.callback.onGenerationStopped()
      }
    }
  }

  companion object {
    const val MESSAGE_LOAD_MODEL = 1
    const val MESSAGE_GENERATE = 2
  }
}
