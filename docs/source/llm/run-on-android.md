# Running LLMs on Android

ExecuTorch's LLM-specific runtime components provide experimental Java APIs, callable from Java or Kotlin, around the core C++ LLM runtime. These APIs are available through the `executorch-android` AAR.

## Prerequisites

Make sure you have a model and tokenizer files ready, as described in the prerequisites section of the [Running LLMs with C++](run-with-c-plus-plus.md) guide.

To add the `executorch-android` library to your app, see [Using ExecuTorch on Android](../using-executorch-android.md). The LLM runner classes are bundled inside the same AAR as the generic `Module` API.

## Runtime API

Once the `executorch-android` AAR is on your classpath, you can import the LLM runner classes from the `org.pytorch.executorch.extension.llm` package. The runner is callable from both Java and Kotlin; the rest of this guide shows both side by side.

### Importing

Java:
```java
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmCallback;

// Only needed for the multimodal ByteBuffer paths in the Images section.
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
```

Kotlin:
```kotlin
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmCallback

// Only needed for the multimodal ByteBuffer paths in the Images section.
import java.nio.ByteBuffer
import java.nio.ByteOrder
```

### LlmModule

The `LlmModule` class provides a simple interface, usable from Java and Kotlin, for loading a text-generation model, configuring its tokenizer, generating token streams, and stopping execution. It also supports multimodal models that accept image and audio inputs alongside a text prompt.

This API is experimental and subject to change.

#### Initialization

Create an `LlmModule` by specifying paths to your serialized model (`.pte`) and tokenizer files. For text-only models, the simple constructor is enough:

Java:
```java
LlmModule module = new LlmModule(
    "/data/local/tmp/llama-3.2-instruct.pte",
    "/data/local/tmp/tokenizer.model",
    0.8f);
```

Kotlin:
```kotlin
val module = LlmModule(
    "/data/local/tmp/llama-3.2-instruct.pte",
    "/data/local/tmp/tokenizer.model",
    0.8f
)
```

For finer control (multimodal model type, BOS/EOS handling, supplementary data files, load mode), use `LlmModuleConfig` with the fluent builder:

Java:
```java
LlmModuleConfig config = LlmModuleConfig.create()
    .modulePath("/data/local/tmp/llama-3.2-instruct.pte")
    .tokenizerPath("/data/local/tmp/tokenizer.model")
    .temperature(0.8f)
    .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
    .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
    .build();

LlmModule module = new LlmModule(config);
```

Kotlin:
```kotlin
val config = LlmModuleConfig.create()
    .modulePath("/data/local/tmp/llama-3.2-instruct.pte")
    .tokenizerPath("/data/local/tmp/tokenizer.model")
    .temperature(0.8f)
    .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
    .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
    .build()

val module = LlmModule(config)
```

Available load modes are `LOAD_MODE_FILE`, `LOAD_MODE_MMAP` (default), `LOAD_MODE_MMAP_USE_MLOCK`, and `LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS`. Available model types are `MODEL_TYPE_TEXT` and `MODEL_TYPE_TEXT_VISION` (the `MODEL_TYPE_MULTIMODAL` constant is currently an alias for `MODEL_TYPE_TEXT_VISION` and selects the same runtime path).

Construction itself is lightweight and does not load the program data immediately.

#### Loading

Explicitly load the model before generation to avoid paying the load cost during your first `generate` call.

Java:
```java
int status = module.load();
if (status != 0) {
  // Handle load failure (status is an ExecuTorch runtime error code).
}
```

Kotlin:
```kotlin
val status = module.load()
if (status != 0) {
    // Handle load failure (status is an ExecuTorch runtime error code).
}
```

If you skip this step, the model is loaded lazily on the first `generate` call.

#### Generating

Generate tokens from a text prompt by passing an `LlmCallback` that receives each token as it is produced. The same callback also receives a JSON-encoded statistics string when generation completes.

Java:
```java
LlmCallback callback = new LlmCallback() {
  @Override
  public void onResult(String token) {
    // Called once per generated token. Append to your UI buffer here.
    System.out.print(token);
  }

  @Override
  public void onStats(String statsJson) {
    // Called once when generation finishes. See extension/llm/runner/stats.h
    // for the field definitions.
    System.out.println("\n" + statsJson);
  }

  @Override
  public void onError(int errorCode, String message) {
    // Called if the runtime reports an error during generation.
  }
};

module.generate("Once upon a time", callback);
```

Kotlin:
```kotlin
val callback = object : LlmCallback {
    override fun onResult(token: String) {
        // Called once per generated token. Append to your UI buffer here.
        print(token)
    }

    override fun onStats(statsJson: String) {
        // Called once when generation finishes. See extension/llm/runner/stats.h
        // for the field definitions.
        println("\n$statsJson")
    }

    override fun onError(errorCode: Int, message: String) {
        // Called if the runtime reports an error during generation.
    }
}

module.generate("Once upon a time", callback)
```

For full control over generation parameters, use `LlmGenerationConfig`:

Java:
```java
LlmGenerationConfig genConfig = LlmGenerationConfig.create()
    .seqLen(2048)
    .temperature(0.8f)
    .echo(false)
    .build();

module.generate("Once upon a time", genConfig, callback);
```

Kotlin:
```kotlin
val genConfig = LlmGenerationConfig.create()
    .seqLen(2048)
    .temperature(0.8f)
    .echo(false)
    .build()

module.generate("Once upon a time", genConfig, callback)
```

`LlmGenerationConfig` exposes `echo`, `maxNewTokens`, `seqLen`, `temperature`, `numBos`, `numEos`, and `warming`. Defaults match the C++ `GenerationConfig` documented in [Running LLMs with C++](run-with-c-plus-plus.md).

#### Stopping Generation

If you need to interrupt a long-running generation, call `stop()` from another thread (or from inside the `onResult` callback):

Java:
```java
module.stop();
```

Kotlin:
```kotlin
module.stop()
```

Generation also runs synchronously on the calling thread, so make sure you invoke `generate()` off the main thread (for example, on a `HandlerThread` or via a `java.util.concurrent.Executor`).

#### Resetting

To clear the prefilled tokens from the KV cache and reset the start position to 0, call:

Java:
```java
module.resetContext();
```

Kotlin:
```kotlin
module.resetContext()
```

This is the equivalent of `reset()` on the iOS runner and `reset()` on the C++ `IRunner`.

### Multimodal Inputs

For models declared as `MODEL_TYPE_TEXT_VISION` (`MODEL_TYPE_MULTIMODAL` is currently an alias), image and audio data are provided through dedicated prefill methods. After prefilling all modalities, call `generate()` with the text prompt to produce the response.

#### Images

Raw uint8 pixel data in CHW order can be supplied as an `int[]`, or as a direct `ByteBuffer` to avoid JNI array copies:

Java:
```java
// As int[]
int[] pixels = ...;       // length == channels * height * width
module.prefillImages(pixels, /*width=*/336, /*height=*/336, /*channels=*/3);

// As direct ByteBuffer (preferred for large images)
byte[] rawBytes = ...;  // length == channels * height * width
ByteBuffer buffer = ByteBuffer.allocateDirect(3 * 336 * 336);
buffer.put(rawBytes);
// Rewind so the JNI side reads from position 0.
buffer.rewind();
module.prefillImages(buffer, 336, 336, 3);
```

Kotlin:
```kotlin
// As IntArray
val pixels: IntArray = ...       // length == channels * height * width
module.prefillImages(pixels, /* width = */ 336, /* height = */ 336, /* channels = */ 3)

// As direct ByteBuffer (preferred for large images)
val rawBytes: ByteArray = ...  // length == channels * height * width
val buffer = ByteBuffer.allocateDirect(3 * 336 * 336).apply {
    put(rawBytes)
    rewind()
}
module.prefillImages(buffer, 336, 336, 3)
```

Pre-normalized float pixel data is also supported, both as a `float[]` and as a direct `ByteBuffer` in native byte order. The two paths intentionally hit different methods: the `float[]` overload is `prefillImages`, while the `ByteBuffer` path is `prefillNormalizedImage` (the names reflect the underlying JNI bindings and are not interchangeable).

Java:
```java
float[] normalized = ...;  // length == channels * height * width
module.prefillImages(normalized, 336, 336, 3);

ByteBuffer floatBuffer = ByteBuffer
    .allocateDirect(3 * 336 * 336 * Float.BYTES)
    .order(ByteOrder.nativeOrder());
// fill floatBuffer with normalized values, then rewind before the call:
floatBuffer.rewind();
module.prefillNormalizedImage(floatBuffer, 336, 336, 3);
```

Kotlin:
```kotlin
val normalized: FloatArray = ...  // length == channels * height * width
module.prefillImages(normalized, 336, 336, 3)

val floatBuffer: ByteBuffer = ByteBuffer
    .allocateDirect(3 * 336 * 336 * Float.SIZE_BYTES)
    .order(ByteOrder.nativeOrder())
// fill floatBuffer with normalized values, then rewind before the call:
floatBuffer.rewind()
module.prefillNormalizedImage(floatBuffer, 336, 336, 3)
```

#### Audio

Preprocessed audio features (for example mel spectrograms produced by a Whisper preprocessor) can be supplied as `byte[]` or `float[]`:

Java:
```java
module.prefillAudio(features, /*batchSize=*/1, /*nBins=*/128, /*nFrames=*/3000);
```

Kotlin:
```kotlin
module.prefillAudio(features, /* batchSize = */ 1, /* nBins = */ 128, /* nFrames = */ 3000)
```

Raw audio samples can be supplied with `prefillRawAudio`:

Java:
```java
module.prefillRawAudio(samples, /*batchSize=*/1, /*nChannels=*/1, /*nSamples=*/16000);
```

Kotlin:
```kotlin
module.prefillRawAudio(samples, /* batchSize = */ 1, /* nChannels = */ 1, /* nSamples = */ 16000)
```

#### Generating with Multimodal Prefill

After prefilling each modality, run `generate()` with the text prompt as usual:

Java:
```java
module.prefillImages(pixels, 336, 336, 3);
module.generate("What's in this image?", callback);
```

Kotlin:
```kotlin
module.prefillImages(pixels, 336, 336, 3)
module.generate("What's in this image?", callback)
```

For text-vision models, a convenience overload accepts the image and prompt together:

Java:
```java
module.generate(
    pixels, /*width=*/336, /*height=*/336, /*channels=*/3,
    "What's in this image?",
    /*seqLen=*/768,
    callback,
    /*echo=*/false);
```

Kotlin:
```kotlin
module.generate(
    pixels, /* width = */ 336, /* height = */ 336, /* channels = */ 3,
    "What's in this image?",
    /* seqLen = */ 768,
    callback,
    /* echo = */ false
)
```

## Demo

See the [Llama Android demo app](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android/LlamaDemo) in `executorch-examples` for an end-to-end project that wires `LlmModule`, `LlmCallback`, and a `HandlerThread` into a chat UI.
