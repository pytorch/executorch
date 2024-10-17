# Summary
This example demonstrates how to run a [llama models](https://www.llama.com/) on mobile via ExecuTorch. We use XNNPACK to accelerate the performance and 4-bit groupwise PTQ quantization to fit the model on a phone.

Here are supported models:

- Llama 3.2 1B and 3B
- Llama 3.1 8B
- Llama 3 8B
- Llama 2 7B

Pretrained models are not included in this repo. Users are suggested to download them [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

# What is Llama?
Llama is a collection of large language models that use publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.

Llama models have shown to perform well on a variety of natural language processing tasks, including language translation, question answering, and text summarization and are also capable of generating human-like text, making Llama models a useful tool for creative writing and other applications where natural language generation is important.

Overall, Llama models are powerful and versatile language models that can be used for a wide range of natural language processing tasks. The model’s ability to generate coherent and contextually relevant text makes it particularly useful for applications such as chatbots, virtual assistants, and language translation.

Please note that the models are subject to the [Llama 2 Acceptable Use Policy](https://github.com/facebookresearch/llama/blob/main/USE_POLICY.md), [Llama 3 Acceptable Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md) and [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).


# Results

Since Llama 2 7B or Llama 3 8B model needs at least 4-bit quantization to fit even within some of the highend phones, results presented here correspond to 4-bit groupwise post-training quantized model.

For Llama 3.2 1B/3B, we validated the models by running them in their original bf16 datatype and unquantized on both Android and iOS phones. The 3B version required high-end phones with larger RAMs to fit the model.

Additionally, 1B/3B models are sensitive to accuracy loss when regular PTQ quantization is applied, so we employed 4bit quantization using [SpinQuant](https://github.com/facebookresearch/SpinQuant/tree/main) to achieve a good balance between accuracy, performance and memory.

<table>
  <tr>
    <td>
      <img src="./llama_via_xnnpack.gif" width="300">
      <br>
      <em>
      Llama3.1 8B, 4bit quantized on Android phone
      </em>
    </td>
    <td><img src="./Android3_2_1B_bf16.gif" width="300">
    <br>
    <em> Llama3.2 1B, unquantized, bf16 on Android phone. </em>
    </td>
  </tr>
</table>

## Quantization:
We employed 4-bit groupwise per token dynamic quantization of all the linear layers of the model. Dynamic quantization refers to quantizating activations dynamically, such that quantization parameters for activations are calculated, from min/max range, at runtime. Here we quantized activations with 8bits (signed integer). Furthermore, weights are statically quantized. In our case weights were per-channel groupwise quantized with 4bit signed integer. For more information refer to this [page](https://github.com/pytorch/ao).

We evaluated WikiText perplexity using [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness). Please note that LM Eval reports perplexity normalized by word count instead of token count. You may see different perplexity for WikiText from other sources if they implement it differntly. More details could be found [here](https://github.com/EleutherAI/lm-evaluation-harness/issues/2301).

Below are the results for two different groupsizes, with max_seq_length 2048, and limit 1000.

|Model | Baseline (FP32) | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------|-----------------| ---------------------- | ---------------
|Llama 2 7B | 9.2 | 10.2 | 10.7
|Llama 3 8B | 7.9 | 9.4 | 9.7

Note that groupsize less than 128 was not enabled, since such models were still too large. This is because our current efforts have focused on enabling FP32 and support for FP16 is under way. What this implies for model size is that 1) embedding table is in FP32 and 2) quantized weights scales are FP32.

### SpinQuant for Llama 3.2 1B/3B models (Optional)

To improve accuracy, we can use [SpinQuant](https://github.com/facebookresearch/SpinQuant/tree/main), a post-training quantization (PTQ) technique that generates new quantized weights. In the standard PTQ process, quantization may lead to a decrease in accuracy when there are outliers. The SpinQuant method takes the original weights and produces optimized quantized weights with minimal outliers, resulting in higher accuracy. This can be achieved without any finetuning of the weights and only requires 100 iterations on a single A100 node.

SpinQuant can generate quantized weights that are [compatible with ExecuTorch](https://github.com/facebookresearch/SpinQuant/tree/main?tab=readme-ov-file#3-export-to-executorch), specifically, it can be integrated with the existing optimized XNNPACK kernels (e.g., group-wise 4bit weight and 8bit dynamic activation). This allows developers to benefit from the higher accuracy of SpinQuant while also taking advantage of the strong performance of ExecuTorch acceleration. We enabled SpinQuant for Llama3.2 1B/3B models on ExecuTorch.

<p align="center">
      <img src="./Android3_2_3B_SpinQuant.gif" width=300>
      <br>
      <em>
      Running Llama3.2 3B on Android phone.
      </em>
      <br>
      <em>
      4bit quantization using SpinQuant
      </em>
</p>

## Enablement

For Llama 3 8B and Llama3.1 8B, we have verified so far on iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S24+ and OnePlus 12 (with 16GB RAM).

We have verified running Llama 2 7B [mobile applications](#step-6-build-mobile-apps) efficiently on select devices including the iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S22 and S24, and OnePlus 12.

## Performance

### Llama 3.2 1B and 3B
Llama 3.2 1B and 3B performance was measured on the OnePlus 12 device. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-5-run-benchmark-on) for generating 128 tokens.

|Model  | bf16 | 4bit(*) via SpinQuant
|--------| ---------------------- | ---------------
|1B  | 19.4 tokens/second | 53.41 tokens/second |
|3B | 7.76 tokens/second | 22.98 tokens/second |

(*) With SpinQuant, we currently quantize 4-bit groupwise (with groupsize 32) weight, 8bit dynamic activation of all the linear layers of the model, except embedding and output layers. The embedding and output layers are quantized as 8-bit per-channel weight and 8-bit dynamic activation.

### Llama3 8B and Llama3.1 8B
Llama 3 8B performance was measured on the Samsung Galaxy S22, S24, and OnePlus 12 devices. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-5-run-benchmark-on).

Note that since Llama3's vocabulary size is 4x that of Llama2, we had to quantize embedding lookup table as well. For these results embedding lookup table was groupwise quantized with 4-bits and group size of 32.

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22  | 7.85 tokens/second | 8.4 tokens/second |
|Galaxy S24 | 10.91 tokens/second | 11.21 tokens/second |
|OnePlus 12 | 10.85 tokens/second | 11.02 tokens/second |

### Llama2 7B
Llama 2 7B performance was measured on the Samsung Galaxy S22, S24, and OnePlus 12 devices. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-5-run-benchmark-on).

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22  | 8.15 tokens/second | 8.3 tokens/second |
|Galaxy S24 | 10.66 tokens/second | 11.26 tokens/second |
|OnePlus 12 | 11.55 tokens/second | 11.6 tokens/second |

# Instructions

## Tested on

- MacOS M1/M2, Linux.
- For Llama 2 7B, your device may require at least 32GB RAM. If this is a constraint for you, please try the smaller stories model.

## Step 1: Setup
> :warning: **double check your python environment**: make sure `conda activate <VENV>` is run before all the bash and python scripts.

1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_requirements.sh --pybind xnnpack`
2. Run `examples/models/llama/install_requirements.sh` to install a few dependencies.


## Step 2: Prepare model

### Option A: Download and export Llama3.2 1B/3B model.

1. Download `consolidated.00.pth`, `params.json` and `tokenizer.model` from [Llama website](https://www.llama.com/llama-downloads/) or [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B). For chat use-cases, download the instruct models.

2. Export model and generate `.pte` file. Use original bfloat16 version, without any quantization.

```
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=path/to/checkpoint.pth
LLAMA_PARAMS=path/to/params.json

python -m examples.models.llama.export_llama \
  --checkpoint "${LLAMA_CHECKPOINT:?}" \
  --params "${LLAMA_PARAMS:?}" \
  -kv \
  --use_sdpa_with_kv_cache \
  -X \
  -d bf16 \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name="llama3_2.pte"
```

Optionally, we can apply SpinQuant to quantize the model without sacrifacing too much accuracy loss.

To use SpinQuant, follow its [instruction](https://github.com/facebookresearch/SpinQuant/tree/main?tab=readme-ov-file#3-export-to-executorch) for exporting checkpoint to ExecuTorch and then export the SpinQuant checkpoint.

```
# Set these paths to point to the exported files
LLAMA_QUANTIZED_CHECKPOINT=path/to/spinquant/checkpoint.pth
LLAMA_PARAMS=path/to/params.json

python -m examples.models.llama.export_llama \
   --checkpoint "${LLAMA_QUANTIZED_CHECKPOINT:?}" \
   --params "${LLAMA_PARAMS:?}" \
   --use_sdpa_with_kv_cache \
   -X \
   --preq_mode 8da4w_output_8da8w \
   --preq_group_size 32 \
   --max_seq_length 2048 \
   --output_name "llama3_2.pte" \
   -kv \
   -d fp32 \
   --preq_embedding_quantize 8,0 \
   --use_spin_quant native \
   --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
```

### Option B: Download and export Llama 3 8B instruct model

You can export and run the original Llama 3 8B instruct model.

1. Llama 3 pretrained parameters can be downloaded from [Meta's official Llama 3 repository](https://github.com/meta-llama/llama3/).

2. Export model and generate `.pte` file
    ```
    python -m examples.models.llama.export_llama --checkpoint <consolidated.00.pth> -p <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w  --group_size 128 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --embedding-quantize 4,32 --output_name="llama3_kv_sdpa_xnn_qe_4_32.pte"
    ```

    Due to the larger vocabulary size of Llama 3, we recommend quantizing the embeddings with `--embedding-quantize 4,32` as shown above to further reduce the model size.

### Option C: Download and export stories110M model

If you want to deploy and run a smaller model for educational purposes. From `executorch` root:

1. Download `stories110M.pt` and `tokenizer.model` from Github.
    ```
    wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
    wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
    ```
2. Create params file.
    ```
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
    ```
3. Export model and generate `.pte` file.
    ```
    python -m examples.models.llama.export_llama -c stories110M.pt -p params.json -X -kv
    ```

### Option D: Download and export Llama 2 7B model

You can export and run the original Llama 2 7B model.

1. Llama 2 pretrained parameters can be downloaded from [Meta's official website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) or from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b).

2. Edit `params.json` file. Replace `"vocab_size": -1` with `"vocab_size": 32000`. This is a short-term workaround.

3. Export model and generate `.pte` file:
    ```
    python -m examples.models.llama.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w --group_size 128 -d fp32
    ```
4. Create tokenizer.bin.
    ```
    python -m extension.llm.tokenizer.tokenizer -t <tokenizer.model> -o tokenizer.bin
    ```

### Option E: Download models from Hugging Face and convert from safetensor format to state dict


You can also download above models from [Hugging Face](https://huggingface.co/). Since ExecuTorch starts from a PyTorch model, a script like below can be used to convert the Hugging Face safetensors format to PyTorch's state dict. It leverages the utils provided by [TorchTune](https://github.com/pytorch/torchtune).


```Python
from torchtune.utils import FullModelHFCheckpointer
from torchtune.models import convert_weights
import torch

# Convert from safetensors to TorchTune. Suppose the model has been downloaded from Hugging Face
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir='/home/.cache/huggingface/hub/models/snapshots/hash-number',
    checkpoint_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'],
    output_dir='/the/destination/dir' ,
    model_type='LLAMA3' # or other types that TorchTune supports
)

print("loading checkpoint")
sd = checkpointer.load_checkpoint()

# Convert from TorchTune to Meta (PyTorch native)
sd = convert_weights.tune_to_meta(sd['model'])

print("saving checkpoint")
torch.save(sd, "/the/destination/dir/checkpoint.pth")
```

## (Optional) Finetuning

If you want to finetune your model based on a specific dataset, PyTorch provides [TorchTune](https://github.com/pytorch/torchtune) - a native-Pytorch library for easily authoring, fine-tuning and experimenting with LLMs.

Once you have [TorchTune installed](https://github.com/pytorch/torchtune?tab=readme-ov-file#get-started) you can finetune Llama2 7B model using LoRA on a single GPU, using the following command. This will produce a checkpoint where the LoRA weights are merged with the base model and so the output checkpoint will be in the same format as the original Llama2 model.

```
tune run lora_finetune_single_device \
--config llama2/7B_lora_single_device \
checkpointer.checkpoint_dir=<path_to_checkpoint_folder>  \
tokenizer.path=<path_to_checkpoint_folder>/tokenizer.model
```

To run full finetuning with Llama2 7B on a single device, you can use the following command.

```
tune run full_finetune_single_device \
--config llama2/7B_full_single_device \
checkpointer.checkpoint_dir=<path_to_checkpoint_folder> \
tokenizer.path=<path_to_checkpoint_folder>/tokenizer.model
```

## Step 3: Evaluate model accuracy

> Forewarning: Model evaluation without a GPU may take a long time, especially on larger models.

We use [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate model accuracy.

For base models, use the following example command to calculate its perplexity based on WikiText.
```
python -m examples.models.llama.eval_llama \
	-c <checkpoint.pth> \
	-p <params.json> \
	-t <tokenizer.model/bin> \
	-kv \
	-d <checkpoint dtype> \
	--max_seq_len <max sequence length> \
	--limit <number of samples>
```

For instruct models, use the following example command to calculate its MMLU score.
```
python -m examples.models.llama.eval_llama \
	-c <checkpoint.pth> \
	-p <params.json> \
	-t <tokenizer.model/bin> \
	-kv \
	-d <checkpoint dtype> \
	--tasks mmlu \
	--num_fewshot 5 \
	--max_seq_len <max sequence length>
```

## Step 4: Run on your computer to validate

1. Build executorch with optimized CPU performance as follows. Build options available [here](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt#L59).
    ```
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -Bcmake-out .

    cmake --build cmake-out -j16 --target install --config Release
    ```
Note for Mac users: There's a known linking issue with Xcode 15.1. Refer to the session of Common Issues and Mitigations below for solutions.

2. Build llama runner.
    ```
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -Bcmake-out/examples/models/llama \
        examples/models/llama

    cmake --build cmake-out/examples/models/llama -j16 --config Release
    ```

3. Run model. Run options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/main.cpp#L18-L40).
    ```
    cmake-out/examples/models/llama/llama_main --model_path=<model pte file> --tokenizer_path=<tokenizer.model> --prompt=<prompt>
    ```

For Llama2 models, pass the converted `tokenizer.bin` file instead of `tokenizer.model`.

To build for CoreML backend and validate on Mac, replace `-DEXECUTORCH_BUILD_XNNPACK=ON` with `-DEXECUTORCH_BUILD_COREML=ON`

## Step 5: Run benchmark on Android phone

**1. Build llama runner binary for Android**

*Pre-requisite*: Android NDK (tested with r27b) which can be downloaded from [here](https://developer.android.com/ndk/downloads). Note that the mac binary can be unpackaged and you can locate NDK folder from it.

**1.1 Set Android NDK**
```
export ANDROID_NDK=<path-to-android-ndk>
```
**1.2 Build executorch and associated libraries for android.**
```
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -Bcmake-out-android .

cmake --build cmake-out-android -j16 --target install --config Release
```

**1.2 Build llama runner for android**
```
cmake  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama

cmake --build cmake-out-android/examples/models/llama -j16 --config Release
```

**2. Run on Android via adb shell**

*Pre-requisite*: Make sure you enable USB debugging via developer options on your phone

**2.1 Connect your android phone**

**2.2 Upload model, tokenizer and llama runner binary to phone**
```
adb shell mkdir -p /data/local/tmp/llama
adb push <model.pte> /data/local/tmp/llama/
adb push <tokenizer.model> /data/local/tmp/llama/
adb push cmake-out-android/examples/models/llama/llama_main /data/local/tmp/llama/
```

**2.3 Run model**
```
adb shell "cd /data/local/tmp/llama && ./llama_main --model_path <model.pte> --tokenizer_path <tokenizer.model> --prompt \"Once upon a time\" --seq_len 120"
```
## Step 6: Build Mobile apps

### iOS

Please refer to [this tutorial](https://pytorch.org/executorch/main/llm/llama-demo-ios.html) to for full instructions on building the iOS LLAMA Demo App. Rename `tokenizer.model` file to `tokenizer.bin` because the demo app looks for the tokenizer file with .bin extension.

### Android
Please refer to [this tutorial](https://pytorch.org/executorch/main/llm/llama-demo-android.html) to for full instructions on building the Android LLAMA Demo App.

## Optional: Smaller models delegated to other backends
Currently we supported lowering the stories model to other backends, including, CoreML, MPS and QNN. Please refer to the instruction
for each backend ([CoreML](https://pytorch.org/executorch/main/build-run-coreml.html), [MPS](https://pytorch.org/executorch/main/build-run-mps.html), [QNN](https://pytorch.org/executorch/main/build-run-qualcomm-ai-engine-direct-backend.html)) before trying to lower them. After the backend library is installed, the script to export a lowered model is

- Lower to CoreML: `python -m examples.models.llama.export_llama -kv --disable_dynamic_shape --coreml -c stories110M.pt -p params.json `
- MPS: `python -m examples.models.llama.export_llama -kv --disable_dynamic_shape --mps -c stories110M.pt -p params.json `
- QNN: `python -m examples.models.llama.export_llama -kv --disable_dynamic_shape --qnn -c stories110M.pt -p params.json `

The iOS LLAMA app supports the CoreML and MPS model and the Android LLAMA app supports the QNN model. On Android, it also allow to cross compiler the llama runner binary, push to the device and run.

For CoreML, there are 2 additional optional arguments:
* `--coreml-ios`: Specify the minimum iOS version to deploy (and turn on available optimizations). E.g. `--coreml-ios 18` will turn on [in-place KV cache](https://developer.apple.com/documentation/coreml/mlstate?language=objc) and [fused scaled dot product attention kernel](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS18.transformers.scaled_dot_product_attention) (the resulting model will then need at least iOS 18 to run, though)
* `--coreml-quantize`: Use [quantization tailored for CoreML](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html). E.g. `--coreml-quantize b4w` will perform per-block 4-bit weight-only quantization in a way tailored for CoreML

# What is coming next?
## Quantization
- Enabling FP16 model to leverage smaller groupsize for 4-bit quantization.
- Enabling GPTQ for 4-bit groupwise quantization
- Enabling custom quantization
- Lower bit quantization
## Models
- Enabling more generative AI models and architectures.
- Enable support for mult-modal models like LlaVa.
## Performance
- Performance improvement via techniques such as speculative decoding
- Enabling LLama2 7b and other architectures via Vulkan
- Enabling performant execution of widely used quantization schemes.


# Notes
This example tries to reuse the Python code, with minimal modifications to make it compatible with current ExecuTorch:
1. Since ExecuTorch does not support complex Tensor data type, use the customized functions to have rotary embedding with real numbers. Please see [GitHub issue: Support complex data type in ExecuTorch](https://github.com/pytorch/executorch/issues/886).
2. No CUDA. ExecuTorch is focused on Edge use cases where CUDA is not available on most of the edge devices.
3. No dependencies on fairscale. The ColumnParallelLinear, ParallelEmbedding and training are not needed and supported in ExecuTorch.


# Common Issues and Mitigations:
- To clean your build:
```
git clean -xfd
pip uninstall executorch
./install_requirements.sh --pybind xnnpack

rm -rf cmake-out
```
- If you encounter `pthread` related issues during link time, add `pthread` in `target_link_libraries` in `CMakeLists.txt`
- On Mac, if there is linking error in Step 4 with error message like
```
0  0x100823648  __assert_rtn + 72
1  0x10074bc5c  ld::Fixup::applyFixup(ld::Atom const*, ld::LayoutLinkedImage const&, unsigned char*) const + 8268
2  0x1007de7d8  ___ZN2ld16LayoutExecutable27writeContentWithoutLinkEditENSt3__14spanIhLm18446744073709551615EEEy_block_invoke + 332
3  0x188cca428  _dispatch_client_callout2 + 20
4  0x188cde850  _dispatch_apply_invoke3 + 336
5  0x188cca3e8  _dispatch_client_callout + 20
6  0x188ccbc68  _dispatch_once_callout + 32
7  0x188cdeeec  _dispatch_apply_invoke_and_wait + 372
8  0x188cdde9c  _dispatch_apply_with_attr_f + 1212
9  0x188cde08c  dispatch_apply + 96
10  0x1007de9e4  void mapReduce<ld::Atom const*, mach_o::Error>(std::__1::span<ld::Atom const*, 18446744073709551615ul>, unsigned long, void (unsigned long, mach_o::Error&, std::__1::span<ld::Atom const*, 18446744073709551615ul>) block_pointer, void (std::__1::span<mach_o::Error, 18446744073709551615ul>) block_pointer) + 336
11  0x1007de594  ld::LayoutExecutable::writeContentWithoutLinkEdit(std::__1::span<unsigned char, 18446744073709551615ul>, unsigned long long) + 1180
12  0x1007e4020  ld::LayoutExecutable::writeToFile(char const*) + 15248
13  0x1007962e8  main + 9424
ld: Assertion failed: (extras.otherInstrOffset != 0 && "Kind::arm64_adrp_ldr missing extra info"), function applyFixup, file Fixup.cpp, line 793.
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```
It's a known issue for Xcode version 15.1.
Mitigation: update to most recent Xcode version, clean and rebuild.
