# Building ExecuTorch Android Demo App for Llama running XNNPack

This tutorial covers the end to end workflow for building an android demo app using CPU on device via XNNPack framework.
More specifically, it covers:
1. Export and quantization of Llama and Llava models against the XNNPack backend.
2. Building and linking libraries that are required to inference on-device for Android platform.
3. Building the Android demo app itself.

Phone verified: OnePlus 12. Samsung S23 (Llama only), Samsung S24+ (Llama only), Pixel 8 Pro (Llama only)


## Known Issues
* With prompts like “What is the maxwell equation” the runner+jni is unable to handle odd unicodes.

## Prerequisites
* Install [Java 17 JDK](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html).
* Install the [Android SDK API Level 34](https://developer.android.com/about/versions/14/setup-sdk) and [Android NDK 25.0.8775105](https://developer.android.com/studio/projects/install-ndk).
* If you have Android Studio set up, you can install them with
  * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Platforms -> Check the row with API Level 34.
  * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Tools -> Check NDK (Side by side) row.
* Alternatively, you can follow [this guide](https://github.com/pytorch/executorch/blob/856e085b9344c8b0bf220a97976140a5b76356aa/examples/demo-apps/android/LlamaDemo/SDK.md) to set up Java/SDK/NDK with CLI.
Supported Host OS: CentOS, macOS Sonoma on Apple Silicon.


Note: This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis), with NDK 25.0.8775105.



## Setup ExecuTorch
In this section, we will need to set up the ExecuTorch repo first with Conda environment management. Make sure you have Conda available in your system (or follow the instructions to install it [here](https://anaconda.org/anaconda/conda)). The commands below are running on Linux (CentOS).

Create a Conda environment
```
conda create -yn executorch python=3.10.0
conda activate executorch
```

Checkout ExecuTorch repo and sync submodules
```
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync
git submodule update --init
```
Install dependencies
```
./install_requirements.sh
```

Optional: Use the --pybind flag to install with pybindings.
```
./install_requirements.sh --pybind xnnpack
```


## Prepare Models
In this demo app, we support text-only inference with up-to-date Llama models and image reasoning inference with LLaVA 1.5.

### For Llama model
* You can download original model weights for Llama through Meta official [website](https://llama.meta.com/), or via Huggingface ([Llama 3.1 8B Instruction](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct))
* For Llama 2 models, Edit params.json file. Replace "vocab_size": -1 with "vocab_size": 32000. This is a short-term workaround
* Run `examples/models/llama2/install_requirements.sh` to install dependencies.
* Export Llama model and generate .pte file

```
python -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w --group_size 128 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --output_name="llama.pte"
```

You may wonder what the ‘--metadata’ flag is doing. This flag helps export the model with proper special tokens added that the runner can detect EOS tokens easily.

* Convert tokenizer for Llama 2
```
python -m extension.llm.tokenizer.tokenizer -t <tokenizer.model> -o tokenizer.bin
```
* Convert tokenizer for Llama 3 - Rename `tokenizer.model` to `tokenizer.bin`.

### For LLaVA model
* For the Llava 1.5 model, you can get it from Huggingface [here](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
* Run `examples/models/llava/install_requirements.sh` to install dependencies.
* Run the following command to generate llava.pte, tokenizer.bin and an image tensor (serialized in TorchScript) image.pt.

```
python -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
```
* You can find more information [here](https://github.com/pytorch/executorch/tree/main/examples/models/llava).


## Pushing Model and Tokenizer
Once you have the model and tokenizer ready, you can push them to the device before we start building the Android demo app.
```
adb shell mkdir -p /data/local/tmp/llama
adb push llama.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

## Build AAR Library
1. Open a terminal window and navigate to the root directory of the executorch
2. Set the following environment variables:
```
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABI=arm64-v8a
```
*Note: <path_to_android_ndk> is the root for the NDK, which is usually under ~/Library/Android/sdk/ndk/XX.Y.ZZZZZ for macOS, and contains NOTICE and README.md. We use <path_to_android_ndk>/build/cmake/android.toolchain.cmake for CMake to cross-compile.*
3. Build the Android Java extension code:
```
pushd extension/android
./gradlew build
popd
```
4. Run the following command set up the required JNI library:
```
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:setup
popd
```
Alternative you can also just run the shell script directly as in the root directory:
```
sh examples/demo-apps/android/LlamaDemo/setup.sh
```

This is running the shell script which configures the required core ExecuTorch, Llama2/3, and Android libraries, builds them, and copies them to jniLibs.

**Output**: The executorch-llama.aar file will be generated in a newly created folder in the example/demo-apps/android/LlamaDemo/app/libs directory. This is the path that the Android app expects it to be in.

**Note**: If you are building the Android app mentioned in the next section on a separate machine (i.e. MacOS but building and exporting on Linux), make sure you copy the aar file generated from setup script to “examples/demo-apps/android/LlamaDemo/app/libs” before building the Android app.

### Alternative: Use prebuilt AAR library
1. Open a terminal window and navigate to the root directory of the executorch.
2. Run the following command to download the prebuilt library
```
bash examples/demo-apps/android/LlamaDemo/download_prebuilt_lib.sh
```
The prebuilt AAR library contains the Java library and the JNI binding for NativePeer.java and ExecuTorch native library, including core ExecuTorch runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels, and Quantized kernels. It comes with two ABI variants, arm64-v8a and x86_64.
If you need to use other dependencies (like tokenizer), please build from the local machine option.

## Run the Android Demo App
### Alternative 1: Android Studio (Recommended)
1. Open Android Studio and select “Open an existing Android Studio project” to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R). This builds and launches the app on the phone.

### Alternative 2: Command line
Without Android Studio UI, we can run gradle directly to build the app. We need to set up the Android SDK path and invoke gradle.
```
export ANDROID_HOME=<path_to_android_sdk_home>
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:installDebug
popd
```
If the app successfully run on your device, you should see something like below:

<p align="center">
<img src="../screenshots/opening_the_app_details.png" width=800>
</p>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
