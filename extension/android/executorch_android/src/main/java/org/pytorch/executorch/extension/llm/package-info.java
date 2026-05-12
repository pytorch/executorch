/**
 * ExecuTorch LLM extension for Android.
 *
 * <p>This package provides Java bindings for running large language models (LLMs)
 * on Android using ExecuTorch. It supports text generation, tokenization,
 * and streaming token callbacks.
 *
 * <h2>Quick Start</h2>
 *
 * <pre>{@code
 * import org.pytorch.executorch.extension.llm.LlmModule;
 *
 * // Load a Llama model
 * LlmModule llm = new LlmModule(
 *     "/data/local/tmp/llama.pte",
 *     "/data/local/tmp/tokenizer.bin",
 *     0.8f
 * );
 * llm.load();
 *
 * // Generate text token by token
 * llm.generate("Hello, my name is", 200, new LlmCallback() {
 *     public void onResult(String token) {
 *         System.out.print(token);
 *     }
 *     public void onStats(String stats) {
 *         System.out.println("\nStats: " + stats);
 *     }
 * });
 * }</pre>
 *
 * <h2>Key Classes</h2>
 *
 * <ul>
 *   <li>{@link org.pytorch.executorch.extension.llm.LlmModule} — load and run an LLM</li>
 *   <li>{@link org.pytorch.executorch.extension.llm.LlmModuleConfig} — configure model paths and settings</li>
 *   <li>{@link org.pytorch.executorch.extension.llm.LlmGenerationConfig} — control generation (temperature, seq length)</li>
 * </ul>
 *
 * <h2>More Resources</h2>
 *
 * <ul>
 *   <li><a href="https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/LlamaDemo">
 *       Llama Android Demo App</a> — full working app with UI</li>
 *   <li><a href="https://pytorch.org/executorch/main/using-executorch-android.html">
 *       Using ExecuTorch on Android</a></li>
 * </ul>
 */
package org.pytorch.executorch.extension.llm;
