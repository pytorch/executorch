/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.jni.HybridData
import com.facebook.jni.annotations.DoNotStrip
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.Closeable
import java.util.concurrent.locks.ReentrantLock
import org.pytorch.executorch.annotations.Experimental

/**
 * Java wrapper for ExecuTorch Module.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
open class Module private constructor(moduleAbsolutePath: String, loadMode: Int, numThreads: Int) :
    Closeable {

  private val mHybridData: HybridData
  private val mMethodMetadata: Map<String, MethodMetadata>

  /** Lock protecting the non-thread safe methods in mHybridData. */
  private val mLock = ReentrantLock()

  init {
    ExecuTorchRuntime.getRuntime()
    mHybridData = initHybrid(moduleAbsolutePath, loadMode, numThreads)
    mMethodMetadata = populateMethodMeta()
  }

  private fun populateMethodMeta(): Map<String, MethodMetadata> {
    val methods = getMethodsNative()
    val metadata = HashMap<String, MethodMetadata>()
    for (name in methods) {
      metadata[name] = MethodMetadata(name, getUsedBackends(name))
    }
    return metadata
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the ExecuTorch module's 'forward' method. Note: if method 'forward'
   *   requires inputs but no inputs are given, the function will not error out, but run 'forward'
   *   with sample inputs.
   * @return return value from the 'forward' method.
   */
  open fun forward(vararg inputs: EValue): Array<EValue> = execute("forward", *inputs)

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value from the method.
   */
  open fun execute(methodName: String, vararg inputs: EValue): Array<EValue> {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return executeNative(methodName, *inputs)
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip
  private external fun executeNative(methodName: String, vararg inputs: EValue): Array<EValue>

  /**
   * Load a method on this module. This might help with the first time inference performance,
   * because otherwise the method is loaded lazily when it's execute. Note: this function is
   * synchronous, and will block until the method is loaded. Therefore, it is recommended to call
   * this on a background thread. However, users need to make sure that they don't execute before
   * this function returns.
   */
  open fun loadMethod(methodName: String) {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      val errorCode = loadMethodNative(methodName)
      if (errorCode != 0) {
        throw ExecutorchRuntimeException(errorCode, "Failed to load method: $methodName")
      }
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun loadMethodNative(methodName: String): Int

  /**
   * Returns the names of the backends in a certain method.
   *
   * @param methodName method name to query
   * @return an array of backend name
   */
  @DoNotStrip private external fun getUsedBackends(methodName: String): Array<String>

  /**
   * Returns the names of methods.
   *
   * @return name of methods in this Module
   */
  open fun getMethods(): Array<String> {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return getMethodsNative()
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun getMethodsNative(): Array<String>

  /**
   * Get the corresponding [MethodMetadata] for a method
   *
   * @param name method name
   * @return [MethodMetadata] for this method
   */
  open fun getMethodMetadata(name: String): MethodMetadata {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return mMethodMetadata[name]
          ?: throw IllegalArgumentException("method $name does not exist for this module")
    } finally {
      mLock.unlock()
    }
  }

  /** Retrieve the in-memory log buffer, containing the most recent ExecuTorch log entries. */
  open fun readLogBuffer(): Array<String>? {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return readLogBufferNative()
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun readLogBufferNative(): Array<String>?

  /**
   * Dump the ExecuTorch ETRecord file to /data/local/tmp/result.etdump.
   *
   * Currently for internal (minibench) use only.
   *
   * @return true if the etdump was successfully written, false otherwise.
   */
  @Experimental
  open fun etdump(): Boolean {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return etdumpNative()
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun etdumpNative(): Boolean

  /**
   * Dump the ExecuTorch ETDump file to [outputPath].
   *
   * @param outputPath absolute path to write the etdump file to.
   * @return true if the etdump was successfully written, false otherwise.
   */
  @Experimental
  open fun etdump(outputPath: String): Boolean {
    mLock.lock()
    try {
      check(mHybridData.isValid) { "Module has been destroyed" }
      return etdumpToNative(outputPath)
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun etdumpToNative(outputPath: String): Boolean

  /**
   * Explicitly destroys the native Module object. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling `destroy` can free memory more
   * quickly. See [com.facebook.jni.HybridData.resetNative].
   */
  open fun destroy() {
    if (mLock.tryLock()) {
      try {
        if (mHybridData.isValid) {
          mHybridData.resetNative()
        }
      } finally {
        mLock.unlock()
      }
    } else {
      throw IllegalStateException("Cannot destroy module while method is executing")
    }
  }

  override fun close() {
    destroy()
  }

  companion object {
    init {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(SystemDelegate())
      }
      NativeLoader.loadLibrary("executorch")
    }

    /** Load mode for the module. Load the whole file as a buffer. */
    const val LOAD_MODE_FILE = 0

    /** Load mode for the module. Use mmap to load pages into memory. */
    const val LOAD_MODE_MMAP = 1

    /** Load mode for the module. Use memory locking and handle errors. */
    const val LOAD_MODE_MMAP_USE_MLOCK = 2

    /** Load mode for the module. Use memory locking and ignore errors. */
    const val LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3

    /**
     * Loads a serialized ExecuTorch module from the specified path on the disk.
     *
     * @param modelPath path to file that contains the serialized ExecuTorch module.
     * @param loadMode load mode for the module. See constants in [Module].
     * @param numThreads the number of threads to use for inference. A value of 0 defaults to a
     *   hardware-specific default.
     * @return new [Module] object which owns the model module.
     */
    @JvmStatic
    @JvmOverloads
    fun load(modelPath: String?, loadMode: Int = LOAD_MODE_FILE, numThreads: Int = 0): Module {
      ExecuTorchRuntime.validateFilePath(modelPath, "model path")
      return Module(modelPath!!, loadMode, numThreads)
    }

    @DoNotStrip
    @JvmStatic
    private external fun initHybrid(
        moduleAbsolutePath: String,
        loadMode: Int,
        numThreads: Int,
    ): HybridData

    @DoNotStrip @JvmStatic fun readLogBufferStatic(): Array<String>? = readLogBufferStaticNative()

    @DoNotStrip @JvmStatic private external fun readLogBufferStaticNative(): Array<String>?
  }
}
