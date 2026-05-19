/**
 * ExecuTorch Android Java API.
 *
 * <p>This package provides Java bindings for running ExecuTorch models on Android. Use these
 * classes to load a {@code .pte} model file and run inference directly from your Java or Kotlin
 * Android app — no C++ required.
 *
 * <h2>Quick Start</h2>
 *
 * <p><b>Step 1.</b> Add the dependency to your {@code app/build.gradle.kts}:
 *
 * <pre>{@code
 * dependencies {
 *     implementation("org.pytorch:executorch-android:${executorch_version}")
 * }
 * }</pre>
 *
 * <p><b>Step 2.</b> Load your model and run inference:
 *
 * <pre>{@code
 * import org.pytorch.executorch.EValue;
 * import org.pytorch.executorch.Module;
 * import org.pytorch.executorch.Tensor;
 *
 * // Load your exported .pte model file
 * Module module = Module.load("/data/local/tmp/model.pte");
 *
 * // Build an input tensor  e.g. a 1x3x224x224 image
 * float[] inputData = new float[1 * 3 * 224 * 224];
 * Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, 224, 224});
 *
 * // Run inference
 * EValue[] output = module.forward(EValue.from(inputTensor));
 *
 * // Read the result
 * float[] scores = output[0].toTensor().getDataAsFloatArray();
 * }</pre>
 *
 * <h2>Key Classes</h2>
 *
 * <ul>
 *   <li>{@link org.pytorch.executorch.Module} — load and run a {@code .pte} model
 *   <li>{@link org.pytorch.executorch.Tensor} — create input tensors and read outputs
 *   <li>{@link org.pytorch.executorch.EValue} — wrap inputs and unwrap outputs
 *   <li>{@link org.pytorch.executorch.DType} — supported data types (FLOAT, INT32, etc.)
 * </ul>
 *
 * <h2>More Resources</h2>
 *
 * <ul>
 *   <li><a href="https://pytorch.org/executorch/main/using-executorch-android.html">Using
 *       ExecuTorch on Android</a> — full setup guide, AAR install, build from source
 *   <li><a href="https://github.com/meta-pytorch/executorch-examples">Android Demo Apps</a> —
 *       working example apps you can build and run immediately
 * </ul>
 */
package org.pytorch.executorch;
