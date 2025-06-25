# Building ExecuTorch Android Demo App for Llama/Llava running XNNPACK
This tutorial covers the end to end workflow for building an android demo app using CPU on device via XNNPACK framework.
More specifically, it covers:
1. Export and quantization of Llama and Llava models against the XNNPACK backend.
2. Building and linking libraries that are required to inference on-device for Android platform.
3. Building the Android demo app itself.

Phone verified: OnePlus 12, OnePlus 9 Pro. Samsung S23 (Llama only), Samsung S24+ (Llama only), Pixel 8 Pro (Llama only)

## Prerequisites
* Install [Java 17 JDK](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html).
* Install the [Android SDK API Level 34](https://developer.android.com/about/versions/15/setup-sdk) and [Android NDK r27b](https://github.com/android/ndk/releases/tag/r27b).
  * Note: This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis), with NDK 26.3.11579264 and r27b.
* If you have Android Studio set up, you can install them with
  * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Platforms -> Check the row with API Level 34.
  * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Tools -> Check NDK (Side by side) row.
* Alternatively, you can follow [this guide](https://github.com/pytorch/executorch/blob/856e085b9344c8b0bf220a97976140a5b76356aa/examples/demo-apps/android/LlamaDemo/SDK.md) to set up Java/SDK/NDK with CLI.
* Supported Host OS: CentOS, macOS Sonoma on Apple Silicon.


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

## Prepare Models
In this demo app, we support text-only inference with up-to-date Llama models and image reasoning inference with LLaVA 1.5.
* You can request and download model weights for Llama through Meta official [website](https://llama.meta.com/).
* For chat use-cases, download the instruct models instead of pretrained.
* Run `./examples/models/llama/install_requirements.sh` to install dependencies.
* Rename tokenizer for Llama3.x with command: `mv tokenizer.model tokenizer.bin`. We are updating the demo app to support tokenizer in original format directly.

### For Llama 3.2 1B and 3B SpinQuant models
Meta has released prequantized INT4 SpinQuant Llama 3.2 models that ExecuTorch supports on the XNNPACK backend.
* Export Llama model and generate .pte file as below:
```
python -m extension.llm.export.export_llm base.model_class="llama3_2" base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True model.dtype_override="fp32" backend.xnnpack.extended_ops=True base.preq_mode="preq_8da4w_out_8da8w" base.preq_group_size=32 export.max_seq_length=2048 export.max_context_length=2048 base.preq_embedding_quantize=\'8,0\' quantization.use_spin_quant="native" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="llama3_2_spinquant.pte"
```
For convenience, an [exported ExecuTorch SpinQuant model](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/blob/main/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/blob/main/Export_Recipe_Llama_3_2_1B_Instruct_SpinQuant_INT4_EO8.ipynb).

### For Llama 3.2 1B and 3B QAT+LoRA models
Meta has released prequantized INT4 QAT+LoRA Llama 3.2 models that ExecuTorch supports on the XNNPACK backend.
* Export Llama model and generate .pte file as below:
```
python -m extension.llm.export.export_llm base.model_class="llama3_2" base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> quantization.use_qat=True base.use_lora=16 model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True model.dtype_override="fp32" backend.xnnpack.extended_ops=True base.preq_mode="preq_8da4w_out_8da8w" base.preq_group_size=32 export.max_seq_length=2048 export.max_context_length=2048 base.preq_embedding_quantize=\'8,0\' base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="llama3_2_qat_lora.pte"
```
For convenience, an [exported ExecuTorch QAT+LoRA model](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET/blob/main/Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET/blob/main/Export_Recipe_Llama_3_2_1B_Instruct_QLORA_INT4_EO8.ipynb).


### For Llama 3.2 1B and 3B BF16 models
We have supported BF16 as a data type on the XNNPACK backend for Llama 3.2 1B/3B models.
* The 1B model in BF16 format can run on mobile devices with 8GB RAM. The 3B model will require 12GB+ RAM.
* Export Llama model and generate .pte file as below:

```
python -m extension.llm.export.export_llm base.model_class="llama3_2" base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True model.dtype_override="bf16" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="llama3_2_bf16.pte"
```
For convenience, an [exported ExecuTorch bf16 model](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/llama3_2-1B.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/ExportRecipe_1B.ipynb).

For more detail using Llama 3.2 lightweight models including prompt template, please go to our official [website](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-lightweight-models-(1b/3b)-).


### For Llama Guard 1B models
To safeguard your application, you can use our Llama Guard models for prompt classification or response classification as mentioned [here](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/).
* Llama Guard 3-1B is a fine-tuned Llama-3.2-1B pretrained model for content safety classification. It is aligned to safeguard against the [MLCommons standardized hazards taxonomy](https://arxiv.org/abs/2404.12241).
* You can download the latest Llama Guard 1B INT4 model, which is already exported for ExecuTorch, using instructions from [here](https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard3). This model is pruned and quantized to 4-bit weights using 8da4w mode and reduced the size to <450MB to optimize deployment on edge devices.
* You can use the same tokenizer from Llama 3.2.
* To try this model, choose Model Type as LLAMA_GUARD_3 in the demo app below and try prompt classification for a given user prompt.
* We prepared this model using the following command

```
python -m extension.llm.export.export_llm base.checkpoint=<path-to-pruned-llama-guard-1b-checkpoint.pth> base.params=<path-to-your-params.json> model.dtype_override="fp32" model.use_kv_cache=True model.use_sdpa_with_kv_cache=True quantization.qmode="8da4w" quantization.group_size=256 backend.xnnpack.enabled=True export.max_seq_length=8193 export.max_context_length=8193 quantization.embedding_quantize=\'4,32\' base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' base.output_prune_map=<path-to-your-llama_guard-pruned-layers-map.json> export.output_name="llama_guard_3_1b_pruned_xnnpack.pte"
```


### For Llama 3.1 and Llama 2 models
* For Llama 2 models, Edit params.json file. Replace "vocab_size": -1 with "vocab_size": 32000. This is a short-term workaround.
* The Llama 3.1 and Llama 2 models (8B and 7B) can run on devices with 12GB+ RAM.
* Export Llama model and generate .pte file as below:

```
python -m extension.llm.export.export_llm base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True quantization.qmode="8da4w" quantization.group_size=128 model.dtype_override="fp32" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="llama.pte"
```

You may wonder what the ‘--metadata’ flag is doing. This flag helps export the model with proper special tokens added that the runner can detect EOS tokens easily.

* Convert tokenizer for Llama 2 and Llava (skip this for Llama 3.x)
```
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin
```

### For LLaVA model
* For the Llava 1.5 model, you can get it from Huggingface [here](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
* Run `examples/models/llava/install_requirements.sh` to install dependencies.
* Run the following command to generate llava.pte, tokenizer.bin and download an image basketball.jpg.

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
```sh
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABIS=arm64-v8a
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
sh examples/demo-apps/android/LlamaDemo/setup.sh
```

This is running the shell script which configures the required core ExecuTorch, Llama2/3, and Android libraries, builds them into AAR, and copies it to the app.

**Output**: The executorch.aar file will be generated in a newly created folder in the example/demo-apps/android/LlamaDemo/app/libs directory. This is the path that the Android app expects it to be in.

**Note**: If you are building the Android app mentioned in the next section on a separate machine (i.e. MacOS but building and exporting on Linux), make sure you copy the aar file generated from setup script to “examples/demo-apps/android/LlamaDemo/app/libs” before building the Android app.

### Alternative: Use prebuilt AAR library
1. Open a terminal window and navigate to the root directory of the executorch.
2. Run the following command to download the prebuilt library
```
bash examples/demo-apps/android/LlamaDemo/download_prebuilt_lib.sh
```
The prebuilt AAR library contains the Java library and the JNI binding for Module.java and ExecuTorch native library, including core ExecuTorch runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels, and Quantized kernels. It comes with two ABI variants, arm64-v8a and x86_64.
If you need to use other dependencies (like tokenizer), please build from the local machine option.

## Run the Android Demo App
### Alternative 1: Android Studio (Recommended)
1. Open Android Studio and select “Open an existing Android Studio project” to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R). This builds and launches the app on the phone.

### Alternative 2: Command line
Without Android Studio UI, we can run gradle directly to build the app. We need to set up the Android SDK path and invoke gradle.
```
export ANDROID_SDK=<path_to_android_sdk_home>
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
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
