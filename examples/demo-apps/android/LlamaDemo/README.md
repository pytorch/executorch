# Building ExecuTorch LLaMA Android Demo App

This app demonstrates the use of the LLaMA chat app demonstrating local inference use case with ExecuTorch.

## Prerequisites
* Refer to [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment.
* Download and install [Android Studio and SDK](https://developer.android.com/studio).
* Supported Host OS: CentOS, macOS Sonoma on Apple Silicon.

```{note}
This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis), with NDK 25.
```

## Getting models
Please refer to the [ExecuTorch Llama2 docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md) to export the model.

After you export the model and generate tokenizer.bin, push them device:
```bash
adb shell mkdir -p /data/local/tmp/llama
adb push llama2.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

```{note}
The demo app searches in `/data/local/tmp/llama` for .pte and .bin files as LLAMA model and tokenizer.
```

## Build JNI library
1. Open a terminal window and navigate to the root directory of the `executorch`.
2. Set the following environment variables:
```bash
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABI=arm64-v8a
```
3. Run the following command set up the required JNI library:
```bash
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:setup
popd
```
This is running the shell script [setup.sh](./setup.sh) which configures the required core ExecuTorch, LLAMA2, and Android libraries, builds them, and copy to jniLibs.

## Build Java app
### Alternative 1: Android Studio (Recommended)
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R). This builds and launches the app on the phone.

### Alternative 2: Command line
Without Android Studio UI, we can run gradle directly to build the app. We need to set up the Android SDK path and invoke gradle.
```bash
export ANDROID_HOME=<path_to_android_sdk_home>
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:installDebug
popd
```

On the phone or emulator, you can try running the model:
<img src="../_static/img/android_llama_app.png" alt="Android LLaMA App" /><br>

## Takeaways
Through this tutorial we've learnt how to build the ExecuTorch LLAMA library, and expose it to JNI layer to build the Android app.

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
