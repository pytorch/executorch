# Building ExecuTorch Android Demo for Llama running MediaTek
This tutorial covers the end to end workflow for running Llama 3-8B-instruct inference on MediaTek AI accelerators on an Android device.
More specifically, it covers:
1. Export and quantization of Llama models against the MediaTek backend.
2. Building and linking libraries that are required to inference on-device for Android platform using MediaTek AI accelerators.
3. Loading the needed files on the device and running inference.

Verified on MacOS, Linux CentOS (model export), Python 3.10, Android NDK 25.0.8775105
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

Create a Conda environment
```
conda create -yn et_mtk python=3.10.0
conda activate et_mtk
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
## Setup Environment Variables
### Download Buck2 and make executable
* Download Buck2 from the official [Release Page](https://github.com/facebook/buck2/releases/tag/2024-02-01)
* Create buck2 executable
```
zstd -cdq "<downloaded_buck2_file>.zst" > "<path_to_store_buck2>/buck2" && chmod +x "<path_to_store_buck2>/buck2"
```

### Set Environment Variables
```
export BUCK2=path_to_buck/buck2 # Download BUCK2 and create BUCK2 executable
export ANDROID_NDK=path_to_android_ndk
export NEURON_BUFFER_ALLOCATOR_LIB=path_to_buffer_allocator/libneuron_buffer_allocator.so
```

## Build Backend and MTK Llama Runner
Next we need to build and compile the MTK backend and MTK Llama runner.
```
cd examples/mediatek
./mtk_build_examples.sh
```

This will generate a cmake-android-out folder that will contain a runner executable for inferring with Llama models and another library file:
* `cmake-android-out/examples/mediatek/mtk_llama_executor_runner`
* `cmake-android-out/backends/mediatek/libneuron_backend.so`

## Export Llama Model
MTK currently supports Llama 3 exporting.

### Set up Environment
1. Follow the ExecuTorch set-up environment instructions found on the [Getting Started](https://pytorch.org/executorch/stable/getting-started-setup.html) page
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

## Deploy Files on Device

### Prepare to Deploy
Prior to deploying the files on device, make sure to modify the tokenizer, token embedding, and model file names in  examples/mediatek/executor_runner/run_llama3_sample.sh reflect what was generated during the Export Llama Model step.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/mtk_changes_to_shell_file.png" style="width:600px">
</p>

In addition, create a sample_prompt.txt file with a prompt. This will be deployed to the device in the next step.
* Example content of a sample_prompt.txt file:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Deploy
First, make sure your Android phone’s chipset version is compatible with this demo (MediaTek Dimensity 9300 (D9300)) chip. Once you have the model, tokenizer, and runner generated ready, you can push them and the .so files to the device before we start running using the runner via shell.

```
adb shell mkdir -p /data/local/tmp/llama
adb push examples/mediatek/executor_runner/run_llama3_sample.sh /data/local/tmp/llama
adb push sample_prompt.txt /data/local/tmp/llama
adb push cmake-android-out/examples/mediatek/mtk_llama_executor_runner /data/local/tmp/llama
adb push cmake-android-out/backends/mediatek/libneuron_backend.so /data/local/tmp/llama
adb push libneuron_buffer_allocator.so /data/local/tmp/llama
adb push libneuronusdk_adapter.mtk.so /data/local/tmp/llama
adb push embedding_<model_name>_fp32.bin /data/local/tmp/llama
adb push tokenizer.model /data/local/tmp/llama
```

## Run Demo
At this point we have pushed all the required files on the device and we are ready to run the demo!
```
adb shell

<android_device>:/ $ cd data/local/tmp/llama
<android_device>:/data/local/tmp/llama $ sh run_llama3_sample.sh
```

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/mtk_output.png" style="width:800px">
</p>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
