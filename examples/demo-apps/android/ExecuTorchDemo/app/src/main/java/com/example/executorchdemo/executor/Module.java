package com.example.executorchdemo.executor;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/** Java wrapper for an ExecuTorch program. */
public class Module {

  private INativePeer mNativePeer;

  /**
   * Loads a serialized ExecuTorch program from the specified path on the disk to run on specified
   * device.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch program.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @return new object which owns ExecuTorch program.
   */
  public static Module load(final String modelPath, final Map<String, String> extraFiles) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(modelPath, extraFiles));
  }

  /**
   * Loads a serialized ExecuTorch program from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch program.
   * @return new {@link Module} object which owns ExecuTorch program.
   */
  public static Module load(final String modelPath) {
    return load(modelPath, null);
  }

  Module(INativePeer nativePeer) {
    this.mNativePeer = nativePeer;
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the ExecuTorch program's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public EValue forward(EValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Explicitly destroys the native ExecuTorch program. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    mNativePeer.resetNative();
  }
}
