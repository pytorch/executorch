# Building Llama iOS Demo for XNNPACK Backend

This tutorial covers the end to end workflow for building an iOS demo app using XNNPACK backend on device.
More specifically, it covers:
1. Export and quantization of Llama models against the XNNPACK backend.
2. Building and linking libraries that are required to inference on-device for iOS platform using XNNPACK.
3. Building the iOS demo app itself.

## Prerequisites
* [Xcode 15](https://developer.apple.com/xcode)
* [iOS 17 SDK](https://developer.apple.com/ios)

## Setup ExecuTorch
In this section, we will need to set up the ExecuTorch repo first with Conda environment management. Make sure you have Conda available in your system (or follow the instructions to install it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). The commands below are running on Linux (CentOS).

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
* Install the required packages to export the model:

```
./examples/models/llama/install_requirements.sh
```

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
* The 1B model in BF16 format can run on mobile devices with 8GB RAM (iPhone 15 Pro and later). The 3B model will require 12GB+ RAM and hence will not fit on 8GB RAM phones.
* Export Llama model and generate .pte file as below:

```
python -m extension.llm.export.export_llm base.model_class="llama3_2" base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True model.dtype_override="bf16" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' export.output_name="llama3_2_bf16.pte"
```
For convenience, an [exported ExecuTorch bf16 model](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/llama3_2-1B.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/ExportRecipe_1B.ipynb).

For more detail using Llama 3.2 lightweight models including prompt template, please go to our official [website](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-llama-3.2-lightweight-models-(1b/3b)-).

### For Llama 3.1 and Llama 2 models

Export the model
```
python -m extension.llm.export.export_llm base.checkpoint=<path-to-your-checkpoint.pth> base.params=<path-to-your-params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True quantization.qmode="8da4w" quantization.group_size=128 model.dtype_override="fp32" base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' quantization.embedding_quantize=\'4,32\' export.output_name="llama3_kv_sdpa_xnn_qe_4_32.pte"
```

### For LLaVA model
* For the Llava 1.5 model, you can get it from Huggingface [here](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
* Run `examples/models/llava/install_requirements.sh` to install dependencies.
* Run the following command to generate llava.pte, tokenizer.bin and download an image basketball.jpg.

```
python -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
```
* You can find more information [here](https://github.com/pytorch/executorch/tree/main/examples/models/llava).


## Configure the XCode Project

### 1. Install CMake
Download and open the macOS .dmg installer at https://cmake.org/download and move the Cmake app to /Applications folder.
Install Cmake command line tools:

```
sudo /Applications/CMake.app/Contents/bin/cmake-gui --install
```

### 2. Add ExecuTorch Runtime Package

There are two options to add ExecuTorch runtime package into your XCode project:

- [Recommended] Prebuilt package (via Swift Package Manager)
- Manually build the package locally and link them


### 2.1 [Recommended] Prebuilt package (via Swift Package Manager)

The current XCode project is pre-configured to automatically download and link the latest prebuilt package via Swift Package Manager.

#### (Optional) Changing the prebuilt package version
While we recommended using the latest prebuilt package pre-configured with the XCode project, you can also change the package version manually to your desired version.

Go to Project Navigator, click on LLaMA. `Project --> LLaMA --> Package Dependencies`, and update the package dependencies to any of the available options below:

- Branch --> swiftpm-0.7.0.20250401 (amend to match the latest nightly build)
- Branch --> swiftpm-0.6.0

### 2.2 Manually build the package locally and link them

Note: You should only use this step if the prebuilt package doesn't work for your usecase (For example, you require the latest PRs from main, where there are no pre-built package yet)

If you need to manually build the package, run the following command in your terminal:
```
# Install a compatible version of Buck2
BUCK2_RELEASE_DATE="2024-12-16"
BUCK2_ARCHIVE="buck2-aarch64-apple-darwin.zst"
BUCK2=".venv/bin/buck2"

curl -LO "https://github.com/facebook/buck2/releases/download/${BUCK2_RELEASE_DATE}/${BUCK2_ARCHIVE}"
zstd -cdq "$BUCK2_ARCHIVE" > "$BUCK2" && chmod +x "$BUCK2"
rm "$BUCK2_ARCHIVE"

./scripts/build_apple_frameworks.sh
```

 After the build finishes successfully, the resulting frameworks can be found in the `cmake-out` directory. Copy them to your project and link them against your targets.

The following packages should be linked in your app target `LLaMA` (left side, LLaMA --> General --> select LLaMA under "TARGETS" --> scroll down to "Frameworks, Libraries, and Embedded Content")
- backend_coreml
- backend_mps
- backend_xnnpack
- kernels_llm
- kernels_optimized
- kernels_portable
- kernels_quantized

The following package should be linked in your target app `LLaMARunner` (left side, LLaMA --> General --> select LLaMARunner under "TARGETS" --> scroll down to "Frameworks and Libraries")
- executorch

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_choosing_package.png" alt="iOS LLaMA App Choosing package" style="width:600px">
</p>

If you cannot add the package into your app target (it's greyed out), it might have been linked before. You can verify it by checking it from your target app `(LLaMA / LLaMARunner) --> Build Phases --> Link Binary With Libraries`.



 More details on integrating and Running ExecuTorch on Apple Platforms, check out the detailed guide [here](https://pytorch.org/executorch/main/using-executorch-ios#local-build).

### 3. Configure Build Schemes

The project has two build configurations:
- Debug
- [Recommended] Release

Navigate to `Product --> Scheme --> Edit Scheme --> Info --> Build Configuration` and update the configuration to "Release".

We recommend that you only use the Debug build scheme during development, where you might need to access additional logs. Debug build has logging overhead and will impact inferencing performance, while release build has compiler optimizations enabled and all logging overhead removed.

For more details integrating and Running ExecuTorch on Apple Platforms or building the package locally, checkout this [link](https://pytorch.org/executorch/main/using-executorch-ios).

### 4. Build and Run the project

Click the "play" button on top right of your XCode app, or navigate to `Product --> Run` to build and run the app on your phone.

### 5. Pushing Model and Tokenizer

There are two options to copy the model (.pte) and tokenizer files (.model) to your app, depending on whether you are running it on a simulator or device.

#### 5.1 Copy the model and tokenizer to Simulator
* Drag&drop the model and tokenizer files onto the Simulator window and save them somewhere inside the iLLaMA folder.
* Pick the files in the app dialog, type a prompt and click the arrow-up button.

#### 5.2 Copy the model and tokenizer to Device
* Plug the device into your Mac and open the contents in Finder.
* Navigate to the Files tab and drag & drop the model and tokenizer files onto the iLLaMA folder.
* Wait until the files are copied.

### 6. Try out the app
Open the iLLaMA app, click the settings button at the top left of the app to select the model and tokenizer files. When the app successfully runs on your device, you should see something like below:

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app.jpg" alt="iOS LLaMA App" style="width:300px">
</p>


For Llava 1.5 models, you can select and image (via image/camera selector button) before typing prompt and send button.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_llava.jpg" alt="iOS LLaMA App" style="width:300px">
</p>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
