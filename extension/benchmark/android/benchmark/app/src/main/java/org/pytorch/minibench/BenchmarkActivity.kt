/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Message
import android.system.Os
import com.google.gson.Gson
import java.io.File
import java.io.FileWriter
import java.io.IOException

class BenchmarkActivity : Activity() {

  lateinit var model: File
  var numIter: Int = 0
  var numWarmupIter: Int = 0
  var tokenizerPath: String? = null
  var temperature: Float = 0.8f
  var prompt: String = "The ultimate answer"

  private lateinit var handlerThread: HandlerThread
  private lateinit var handler: BenchmarkHandler

  val results: MutableList<BenchmarkMetric> = mutableListOf()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    try {
      Os.setenv("ADSP_LIBRARY_PATH", applicationInfo.nativeLibraryDir, true)
    } catch (e: android.system.ErrnoException) {
      finish()
      return
    }

    val intent = intent
    val modelDir = File(intent.getStringExtra("model_dir")!!)
    model = modelDir.listFiles()!!.first { it.name.endsWith(".pte") }

    numIter = intent.getIntExtra("num_iter", 50)
    numWarmupIter = intent.getIntExtra("num_warm_up_iter", 10)
    tokenizerPath = intent.getStringExtra("tokenizer_path")
    temperature = intent.getFloatExtra("temperature", 0.8f)
    prompt = intent.getStringExtra("prompt") ?: "The ultimate answer"

    handlerThread = HandlerThread("ModelRunner")
    handlerThread.start()
    handler = BenchmarkHandler(handlerThread.looper, this)

    handler.sendEmptyMessage(BenchmarkHandler.MESSAGE_RUN_BENCHMARK)
  }

  fun writeResult() {
    try {
      FileWriter("${filesDir}/benchmark_results.json").use { writer ->
        writer.write(Gson().toJson(results))
      }
    } catch (e: IOException) {
      e.printStackTrace()
    } finally {
      finish()
    }
  }
}

private class BenchmarkHandler(
    looper: Looper,
    private val activity: BenchmarkActivity,
) : Handler(looper) {

  private val modelRunner = ModelRunner()

  override fun handleMessage(msg: Message) {
    when (msg.what) {
      MESSAGE_RUN_BENCHMARK -> {
        modelRunner.runBenchmark(
            activity.model,
            activity.numWarmupIter,
            activity.numIter,
            activity.results,
        )
        if (activity.tokenizerPath == null) {
          activity.writeResult()
        } else {
          sendEmptyMessage(MESSAGE_LLM_RUN_BENCHMARK)
        }
      }
      MESSAGE_LLM_RUN_BENCHMARK -> {
        LlmBenchmark(
            activity,
            activity.model.path,
            activity.tokenizerPath!!,
            activity.prompt,
            activity.temperature,
            activity.results,
        )
      }
    }
  }

  companion object {
    const val MESSAGE_RUN_BENCHMARK = 1
    const val MESSAGE_LLM_RUN_BENCHMARK = 2
  }
}
