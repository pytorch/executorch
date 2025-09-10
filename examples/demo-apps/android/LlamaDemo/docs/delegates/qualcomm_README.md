# Building ExecuTorch Android Demo App for Llama running Qualcomm

This tutorial covers the end to end workflow for building an android demo app using Qualcomm AI accelerators on device.
More specifically, it covers:
1. Export and quantization of Llama models against the Qualcomm backend.
2. Building and linking libraries that are required to inference on-device for Android platform using Qualcomm AI accelerators.
3. Building the Android demo app itself.

Verified on Linux CentOS, QNN SDK [v2.26](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.26.0.240828.zip), python 3.10, Android SDK r27b.

Phone verified: OnePlus 12, Samsung 24+, Samsung 23

## Prerequisites
* Download and unzip QNN SDK [v2.26](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.26.0.240828.zip)
* Download and unzip Android SDK [r27b](https://developer.android.com/ndk/downloads)
* Android phone with Snapdragon8 Gen3 (SM8650) or Gen2 (SM8550). Gen 1 and lower SoC might be supported but not fully validated.
* Desired Llama model weights in .PTH format. You can download them on HuggingFace ([Example](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)).

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

## Setup QNN
```
# Set these variables correctly for your environment
export ANDROID_NDK_ROOT=$HOME/android-ndk-r27b # Download android SDK and unzip to home directory
export QNN_SDK_ROOT=$HOME/Your-SDK-Root #Folder contains lib
export EXECUTORCH_ROOT=$HOME/repos/executorch
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
cp schema/program.fbs exir/_serialize/program.fbs
cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
```

### Build QNN backend with ExecuTorch
```
./backends/qualcomm/scripts/build.sh --release

cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -Bcmake-out .
cmake --build cmake-out -j16 --target install --config Release
```



### Setup Llama Runner
Next we need to build and compile the Llama runner. This is similar to the requirements for running Llama with XNNPACK.
```
./examples/models/llama/install_requirements.sh

cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -Bcmake-out/examples/models/llama \
    examples/models/llama
cmake --build cmake-out/examples/models/llama -j16 --config Release
```

## Export Llama Model
QNN backend currently supports exporting to these data types: fp32, int4/ int8 with PTQ, int4 with SpinQuant (Llama 3 only).

We also support export for different Qualcomm SoC. We have verified SM8650(V75) and SM8550(V73). To export for different SoC, add "--soc_model SM8550" in your export command. Without setting this flag, the export will default to SM8650.

### Export with PTQ
We support PTQ by default. The entire export may take ~20 minutes (Llama 3.1 8B). However, there is accuracy regression and we are working on improving it.
8B models might need 16GB RAM on the device to run.

Examples:
```
# 4 bits weight only quantize
python -m extension.llm.export.export_llm base.checkpoint="${MODEL_DIR}/consolidated.00.pth" base.params="${MODEL_DIR}/params.json" model.use_kv_cache=True model.enable_dynamic_shape=False backend.qnn.enabled=True backend.qnn.quantization="qnn_16a4w" model.dtype_override="fp32" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="test.pte"
```
If the model is really big, it may require model sharding because the Qualcomm DSP is a 32bit system and has a 4GB size limit . For example for Llama 3 8B models, we need to shard the model into 4, but ExecuTorch still packages it into one PTE file. Here is an example:
```
# 8 bits quantization with 4 shards
python -m extension.llm.export.export_llm base.checkpoint="${MODEL_DIR}/consolidated.00.pth" base.params="${MODEL_DIR}/params.json" model.use_kv_cache=True model.enable_dynamic_shape=False backend.qnn.enabled=True backend.qnn.quantization="qnn_8a8w" model.dtype_override="fp32" backend.qnn.num_sharding=4 base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="test.pte"
```
Note: if you encountered issues below
```
[ERROR] [Qnn ExecuTorch]: Cannot Open QNN library libQnnHtp.so, with error: libc++.so.1: cannot open shared object file: No such file or directory
```

Resolve by:

* Install older QNN such as 2.23 or below and copy it from ${QNN_SDK_ROOT}/lib/x86_64-linux-clang
* Install it with apt-get by yourself
* Install it with script in ${QNN_SDK_ROOT}/bin/check-linux-dependency.sh
You could refer to [QNN SDK document](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html?product=1601111740009302#linux-platform-dependencies)
* Install it with Conda:
```
conda install -c conda-forge libcxx=14.0.0
```

After installment, you will need to check libc++.so.1 in your LD_LIBRARY_PATH or system lib. Refer to this [PR](https://github.com/pytorch/executorch/issues/5120) for more detail.

You may also wonder what the "--metadata" flag is doing. This flag helps export the model with proper special tokens added that the runner can detect EOS tokens easily.

Convert tokenizer for Llama 2
```
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin
```
Rename tokenizer for Llama 3 with command: `mv tokenizer.model tokenizer.bin`. We are updating the demo app to support tokenizer in original format directly.


### Export with Spinquant (Llama 3 8B only)
We also support Llama 3 8B for Spinquant where the accuracy regression is minimal.

Deploying large language models like Llama 3 on-device presents the following challenges:
* The model size is too large to fit in device memory for inference.
* High model loading and inference time.
* Difficulty in quantization.

To address these challenges, we have implemented the following solutions:
* Using --pt2e_quantize qnn_16a4w to quantize activations and weights, thereby reducing the on-disk model size and alleviating memory pressure during inference.
* Using --num_sharding 8 to shard the model into sub-parts.
* Performing graph transformations to convert or decompose operations into more accelerator-friendly operations.
* Using --optimized_rotation_path <path_to_optimized_matrix> to apply R1 and R2 of [Spin Quant](https://github.com/facebookresearch/SpinQuant) to improve accuracy.
* Using --calibration_data "<|start_header_id|>system<|end_header_id|..." to ensure that during the quantization of Llama 3 8B Instruct, the calibration includes special tokens in the prompt template. For more details on the prompt template, refer to the [model card](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) of meta llama3 instruct.

To get the optimized matrix, please refer to [SpinQuant](https://github.com/facebookresearch/SpinQuant) on GitHub. You can download the optimized rotation matrices in the Quantized Models section. Please choose "LLaMA-3-8B/8B_W4A16KV16_lr_1.5_seed_0".

To export Llama 3 8B instruct with the Qualcomm AI Engine Direct Backend, ensure the following:
* The host machine has more than 100GB of memory (RAM + swap space).
* The entire process takes a few hours.
* 8B models might need 16GB RAM on the device to run.
```
# Please note that calibration_data must include the prompt template for special tokens.
python -m extension.llm.export.export_llm base.tokenizer=<path_to_tokenizer.model> base.params=<path_to_params.json> base.checkpoint=<path_to_checkpoint_for_Meta-Llama-3-8B-Instruct> model.use_kv_cache=True backend.qnn.enabled=True backend.qnn.quantization="qnn_16a4w" model.enable_dynamic_shape=False backend.qnn.num_sharding=8 backend.qnn.calibration_tasks="wikitext" backend.qnn.calibration_limit=1 backend.qnn.calibration_seq_length=128 backend.qnn.optimized_rotation_path=<path_to_optimized_matrix> backend.qnn.calibration_data="<|start_header_id|>system<|end_header_id|>\n\nYou are a funny chatbot.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCould you tell me about Facebook?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Pushing Model and Tokenizer

Once you have the model and tokenizer ready, you can push them to the device before we start building the android demo app.
```
adb shell mkdir -p /data/local/tmp/llama
adb push llama-exported.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```



## Build AAR Library
1. Open a terminal window and navigate to the root directory of the executorch
Set the following environment variables:
```sh
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABIS=arm64-v8a
export QNN_SDK_ROOT=<path_to_qnn_sdk>
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

Alternative you can also just run the shell script directly as in the root directory:
```sh
sh examples/demo-apps/android/LlamaDemo/setup-with-qnn.sh
```
This is running the shell script which configures the required core ExecuTorch, Llama2/3, and Android libraries, builds them into AAR, and copies it to the app.
Note: If you are building the Android app mentioned in the next section on a separate machine (i.e. MacOS but building and exporting for QNN backend on Linux), make sure you copy the aar file generated from setup-with-qnn script to "examples/demo-apps/android/LlamaDemo/app/libs" before building the Android app.

6. Set up the correct QNN version in gradle rule
Currently, the gralde rule searches for the property `qnnVersion`. When this variable is defined, it will add QNN runtime library to the dependency. To use it, append the string `qnnVersion=<version>` (ex. `qnnVersion=2.37.0`) to the end of the `gradle.properties` file.

## Run the Android Demo App

First, make sure your Android phone's chipset version is compatible with this demo (SM8650, SM8550). You can find the Qualcomm chipset version here in the [mapping](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html).

If you build and run the setup-with-qnn script on a separate machine rather than where you are building the Android app, make sure you copy the aar file it generated into "examples/demo-apps/android/LlamaDemo/app/libs"

### Alternative 1: Android Studio (Recommended)
Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
Run the app (^R). This builds and launches the app on the phone.

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

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on Github.
