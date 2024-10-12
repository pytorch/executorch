# Using Android prebuilt libraries (AAR)

We provide two prebuilt Android libraries (AAR), `executorch.aar` for generic use case (image/audio processing) and `executorch_llama.aar` for LLAMA use case.

## Contents of libraries
- `executorch.aar`
  - [Java library](https://github.com/pytorch/executorch/tree/release/0.4/extension/android/src/main/java/org/pytorch/executorch)
  - JNI contains the JNI binding for [NativePeer.java](https://github.com/pytorch/executorch/blob/release/0.4/extension/android/src/main/java/org/pytorch/executorch/NativePeer.java) and ExecuTorch native library, including core ExecuTorch runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels, and Quantized kernels.
    - Comes with two ABI variants, arm64-v8a and x86_64.
- `executorch_llama.aar`
  - [Java library](https://github.com/pytorch/executorch/tree/release/0.4/extension/android/src/main/java/org/pytorch/executorch) (Note: it contains the same Java classes as the previous Java, but it does not contain the JNI binding for generic Module/NativePeer Java code).
  - JNI contains the JNI binding for [LlamaModule.java](https://github.com/pytorch/executorch/blob/release/0.4/extension/android/src/main/java/org/pytorch/executorch/LlamaModule.java) and ExecuTorch native library, including core ExecuTorch runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels, Quantized kernels, and LLAMA-specific Custom ops library.
    - Comes with two ABI variants, arm64-v8a and x86_64.

## Downloading AAR
[executorch.aar](https://ossci-android.s3.amazonaws.com/executorch/release/0.4/executorch.aar)
[executorch.aar.sha256sums](https://ossci-android.s3.amazonaws.com/executorch/release/0.4/executorch.aar.sha256sums)

## Using prebuilt libraries

To add the Java library to your app, simply download the AAR, and add it to your gradle build rule.

In your app working directory, such as example executorch/examples/demo-apps/android/LlamaDemo,
```
mkdir -p app/libs
curl https://ossci-android.s3.amazonaws.com/executorch/release/0.4/executorch.aar -o app/libs/executorch-llama.aar
```

And include it in gradle:
```
# app/build.grardle.kts
dependencies {
    implementation(files("libs/executorch-llama.aar"))
}
```

Now you can compile your app with the ExecuTorch Android library.
