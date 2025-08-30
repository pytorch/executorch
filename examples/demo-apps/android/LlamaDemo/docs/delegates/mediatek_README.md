# Building ExecuTorch Android Demo for Llama running MediaTek
This tutorial covers the end to end workflow for running Llama 3-8B-instruct inference on MediaTek AI accelerators on an Android device.
More specifically, it covers:
1. Export and quantization of Llama models against the MediaTek backend.
2. Building and linking libraries that are required to inference on-device for Android platform using MediaTek AI accelerators.
3. Loading the needed model files on the device and using the Android demo app to run inference.

Verified on MacOS, Linux CentOS (model export), Python 3.10, Android NDK 26.3.11579264
Phone verified: MediaTek Dimensity 9300 (D9300) chip.

## Prerequisites
* Download and link the Buck2 build, Android NDK, and MediaTek ExecuTorch Libraries from the MediaTek Backend Readme ([link](https://github.com/pytorch/executorch/tree/main/backends/mediatek/scripts#prerequisites)).
* MediaTek Dimensity 9300 (D9300) chip device
* Desired Llama 3 model weights. You can download them on HuggingFace [Example](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)).
* Download NeuroPilot Express SDK from the [MediaTek NeuroPilot Portal](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress):
  - `libneuronusdk_adapter.mtk.so`: This universal SDK contains the implementation required for executing target-dependent code on the MediaTek chip.
  - `libneuron_buffer_allocator.so`: This utility library is designed for allocating DMA buffers necessary for model inference.
  - `mtk_converter-8.8.0.dev20240723+public.d1467db9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`: This library preprocess the model into a MediaTek representation.
  - `mtk_neuron-8.2.2-py3-none-linux_x86_64.whl`: This library converts the model to binaries.

## Setup ExecuTorch
In this section, we will need to set up the ExecuTorch repo first with Conda environment management. Make sure you have Conda available in your system (or follow the instructions to install it [here](https://anaconda.org/anaconda/conda)). The commands below are running on Linux (CentOS).

Checkout ExecuTorch repo and sync submodules

```
git clone -b viable/strict https://github.com/pytorch/executorch.git && cd executorch
```

Create either a Python virtual environment:

```
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```

Or a Conda environment:

```
conda create -n et_xnnpack python=3.10.0 && conda activate et_xnnpack
```

Install dependencies
```
./install_executorch.sh
```

## Setup Environment Variables
### Download Buck2 and make executable
* Download Buck2 from the official [Release Page](https://github.com/facebook/buck2/releases/tag/2024-02-01)
* Create buck2 executable
```
zstd -cdq "<downloaded_buck2_file>.zst" > "<path_to_store_buck2>/buck2" && chmod +x "<path_to_store_buck2>/buck2"
```

### Set Environment Variables
```
export ANDROID_NDK=path_to_android_ndk
export NEURON_BUFFER_ALLOCATOR_LIB=path_to_buffer_allocator/libneuron_buffer_allocator.so
export NEURON_USDK_ADAPTER_LIB=path_to_usdk_adapter/libneuronusdk_adapter.mtk.so
export ANDROID_ABIS=arm64-v8a
```

## Export Llama Model
MTK currently supports Llama 3 exporting.

### Set up Environment
1. Follow the ExecuTorch set-up environment instructions found on the [Getting Started](https://pytorch.org/executorch/main/getting-started-setup.html) page
2. Set-up MTK AoT environment
```
// Ensure that you are inside executorch/examples/mediatek directory
pip3 install -r requirements.txt

pip3 install mtk_neuron-8.2.2-py3-none-linux_x86_64.whl
pip3 install mtk_converter-8.8.0.dev20240723+public.d1467db9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

This was tested with transformers version 4.40 and numpy version 1.23. If you do not have these version then, use the following commands:
```
pip install transformers==4.40

pip install numpy=1.23
```

### Running Export
Prior to exporting, place the config.json, relevant tokenizer files and .bin or .safetensor weight files in `examples/mediatek/models/llm_models/weights`.

Here is an export example ([details](https://github.com/pytorch/executorch/tree/main/examples/mediatek#aot-flow)):
```
cd examples/mediatek
# num_chunks=4, num_tokens=128, cache_size=512
source shell_scripts/export_llama.sh llama3 "" "" "" alpaca.txt
```

There will be 3 main set of files generated:
* num_chunks*2 pte files: half are for prompt and the other half are for generation. Generation pte files are denoted by “1t” in the file name.
* Token embedding bin file: located in the weights folder where `config.json` is placed (`examples/mediatek/modes/llm_models/weight/<model_name>/embedding_<model_name>_fp32.bin`)
* Tokenizer file: `tokenizer.model` file

Note: Exporting model flow can take 2.5 hours (114GB RAM for num_chunks=4) to complete. (Results may vary depending on hardware)

Before continuing forward, make sure to modify the tokenizer, token embedding, and model paths in the  examples/mediatek/executor_runner/run_llama3_sample.sh.

### Deploy
First, make sure your Android phone’s chipset version is compatible with this demo (MediaTek Dimensity 9300 (D9300)) chip. Once you have the model, tokenizer, and runner generated ready, you can push them and the .so files to the device before we start running using the runner via shell.

```
adb shell mkdir -p /data/local/tmp/et-mtk/ (or any other directory name)
adb push embedding_<model_name>_fp32.bin /data/local/tmp/et-mtk
adb push tokenizer.model /data/local/tmp/et-mtk
adb push <exported_prompt_model_0>.pte /data/local/tmp/et-mtk
adb push <exported_prompt_model_1>.pte /data/local/tmp/et-mtk
...
adb push <exported_prompt_model_n>.pte /data/local/tmp/et-mtk
adb push <exported_gen_model_0>.pte /data/local/tmp/et-mtk
adb push <exported_gen_model_1>.pte /data/local/tmp/et-mtk
...
adb push <exported_gen_model_n>.pte /data/local/tmp/et-mtk
```

## Populate Model Paths in Runner

The Mediatek runner (`examples/mediatek/executor_runner/mtk_llama_runner.cpp`) contains the logic for implementing the function calls that come from the Android app.

**Important!** Currently the model paths are set in the runner-level. Modify the values in `examples/mediatek/executor_runner/llama_runner/llm_helper/include/llama_runner_values.h` to set the model paths, tokenizer path, embedding file path, and other metadata.


## Build AAR Library
1. Open a terminal window and navigate to the root directory of the executorch
2. Set the following environment variables:
```sh
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABIS=arm64-v8a
export NEURON_BUFFER_ALLOCATOR_LIB=<path_to_neuron_buffer_allocator_lib>
```
*Note: <path_to_android_ndk> is the root for the NDK, which is usually under ~/Library/Android/sdk/ndk/XX.Y.ZZZZZ for macOS, and contains NOTICE and README.md. We use <path_to_android_ndk>/build/cmake/android.toolchain.cmake for CMake to cross-compile.*

3. Create a directory to hold the AAR
```sh
mkdir -p aar-out
export BUILD_AAR_DIR=aar-out
```

4. Run the following command to build the AAR:
```sh
sh scripts/build_android_library.sh
```

5. Copy the AAR to the app:
```sh
mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
cp aar-out/executorch.aar examples/demo-apps/android/LlamaDemo/app/libs/executorch.aar
```

If you were to unzip the .aar file or open it in Android Studio, verify it contains the following related to MediaTek backend:
* libneuron_buffer_allocator.so
* libneuronusdk_adapter.mtk.so
* libneuron_backend.so (generated during build)

## Run Demo

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
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/opening_the_app_details.png" style="width:800px">
</p>

Once you've loaded the app on the device:
1. Click on the Settings in the app
2. Select MediaTek from the Backend dropdown
3. Click the "Load Model" button. This will load the models from the Runner

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
