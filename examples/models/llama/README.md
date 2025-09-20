# Summary
This example demonstrates how to run [Llama models](https://www.llama.com/) on mobile via ExecuTorch. We use XNNPACK to accelerate the performance and 4-bit groupwise quantization to fit the model on a phone.

Here are supported models:

- Llama 3.2 1B and 3B
- Llama 3.2 Quantized 1B and 3B
- Llama 3.1 8B
- Llama 3 8B
- [Llama 2 7B](../llama2/README.md)

Pretrained models are not included in this repo. Users are suggested to download them [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

This page contains the basic recipe for running Llama. See [Llama utils page](UTILS.md) page for more advanced use-cases such as fine-tuning and running smaller models for educational purposes.

# What is Llama?
Llama is a collection of large language models that use publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.

Llama models have shown to perform well on a variety of natural language processing tasks, including language translation, question answering, and text summarization and are also capable of generating human-like text, making Llama models a useful tool for creative writing and other applications where natural language generation is important.

Overall, Llama models are powerful and versatile language models that can be used for a wide range of natural language processing tasks. The model’s ability to generate coherent and contextually relevant text makes it particularly useful for applications such as chatbots, virtual assistants, and language translation.

Please note that the models are subject to the [Llama 2 Acceptable Use Policy](https://github.com/facebookresearch/llama/blob/main/USE_POLICY.md), [Llama 3 Acceptable Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md) and [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).


# Results

## Llama 3.2 1B/3B and quantized 1B/3B models

For Llama 3.2 1B/3B models, we have enabled the original BF16 format and quantization to 4-bit, using SpinQuant and QAT+LoRA, for enhanced performance.

The quantized models were optimized primarily for Arm CPU architecture by leveraging XNNPACK and Kleidi AI library. Work is underway to specifically enable quantization on mobile accelerators for Llama 1B/3B.

### Enablement

We have successfully verified performance on the following devices: iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S24+, S22 and OnePlus 12 (featuring 16GB RAM).

Note, the Llama 3.2 3B unquantized BF16 model was only tested on the OnePlus 12, which has sufficient memory (16GB RAM) to support its size requirements.

### Quantization

The 1B/3B models are sensitive to accuracy loss when regular post-training quantization (PTQ) is applied. To achieve a balance between accuracy, performance and memory, we utilized 4-bit quantization, using [SpinQuant](https://github.com/facebookresearch/SpinQuant/tree/main) and QAT+LoRA methods.

Our quantization scheme involves three parts, applicable to both methods:

- We quantize all linear layers in all transformer blocks to a 4-bit groupwise scheme (with a group size of 32) for weights and 8-bit per-token dynamic quantization for activations.
- The classification layer is quantized to 8-bit per-channel for weight and 8-bit per token dynamic quantization for activation.
- We employ an 8-bit per channel quantization for embedding.

We use [torchao](https://github.com/pytorch/ao) library APIs to define these schemes.

#### SpinQuant

The SpinQuant method takes the original weights and produces optimized quantized weights with minimal outliers, resulting in higher accuracy. This can be achieved without any finetuning of the weights and only requires 100 iterations on a single A100 node.

SpinQuant can generate quantized weights that are [compatible with ExecuTorch](https://github.com/facebookresearch/SpinQuant/tree/main?tab=readme-ov-file#3-export-to-executorch), specifically, it can be integrated with the existing optimized XNNPACK kernels (e.g., group-wise 4bit weight and 8bit dynamic activation). This allows developers to benefit from the higher accuracy of SpinQuant while also taking advantage of the strong performance of ExecuTorch acceleration.

#### Quantization-Aware Training and LoRA (QAT+LoRA)

Quantization-Aware Training (QAT) is employed to simulate the effects of quantization during the training of Llama-3.2 models, enabling optimization of their performance in low precision environments. To initialize QAT, BF16 Llama-3.2 model checkpoints obtained after supervised fine-tuning (SFT) are utilized and an additional full round of SFT training with QAT is performed. The backbone of the QAT model is then frozen and another round of SFT is performed with low-rank adaptation (LoRA) adaptors applied to all layers within the transformer block. Meanwhile, the LoRA adaptors' weights and activations are maintained in BF16.

### Accuracy

Please see the [Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) for accuracy evalations.

### Performance

Llama 3.2 1B and 3B performance was measured on Android OnePlus 12 device. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-4-run-benchmark-on-android-phone) with prompt length of 64. It is measured with KleidiAI library. KleidiAI is enabled by default on the XNNPACK Backend for all ARM devices.

|Model  | Decode (tokens/s) | Time-to-first-token (sec) | Prefill (tokens/s) | Model size (PTE file size in MiB) | Memory size (RSS in MiB) |
|-------|------------------:|--------------------------:| ------------------:|----------------------------------:| ------------------------:|
|1B BF16 (baseline) | 19.2 |  1.0 | 60.3  | 2,358 | 3,185 |
|1B SpinQuant | 50.2 (2.6x) | 0.3 (-76.9%) | 260.5 (4.3x) | 1,083 (-54.1%)  | 1,921 (-39.7%) |
|1B QAT+LoRA | 45.8 (2.4x) | 0.3 (-76.0%)  | 252.0 (4.2x) | 1,127 (-52.2%)  | 2,255 (-29.2%) |
|3B BF16 (baseline) | 7.6  | 3.0 | 21.2 | 6,129 | 7,419 |
|3B SpinQuant | 19.7 (2.6x) | 0.7 (-76.4%) | 89.7 (4.2x) | 2,435 (-60.3%) | 3,726 (-49.8%) |
|3B QAT+LoRA | 18.5 (2.4x) | 0.7 (-76.1%) | 88.8 (4.2x) | 2,529 (-58.7%) | 4,060 (-45.3%) |


<table>
  <tr>
    <td>
        <img src="Android3_2_1B_bf16.gif" width="300">
        <br>
        <em> Llama3.2 1B, unquantized, BF16 on Android phone. </em>
    </td>
    <td>
      <img src="Android3_2_3B_SpinQuant.gif" width="300">
      <br>
      <em>
      Llama3.2 3B, 4bit quantized (SpinQuant) on Android phone
      </em>
    </td>
  </tr>
</table>

## Llama 3/3.1 8B
Since Llama 3 8B model needs at least 4-bit quantization to fit even within some of the highend phones, results presented here correspond to 4-bit groupwise post-training quantized (PTQ) model.

### Enablement

For Llama 3 8B and Llama3.1 8B, we have verified so far on iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S24+ and OnePlus 12 (with 16GB RAM) by quantizing to 4bit.

### Quantization

We employed PTQ 4-bit groupwise per token dynamic quantization of all the linear layers of the model. Dynamic quantization refers to quantizating activations dynamically, such that quantization parameters for activations are calculated, from min/max range, at runtime. Here we quantized activations with 8bits (signed integer). Furthermore, weights are statically quantized. In our case weights were per-channel groupwise quantized with 4bit signed integer. Due to Llama3's vocabulary size, we had to quantize embedding lookup table as well. For these results embedding lookup table was groupwise quantized with 4-bits and group size of 32.

We use [torchao](https://github.com/pytorch/ao) library APIs to define these schemes.

### Accuracy

We evaluated WikiText perplexity using [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness). Below are the results for two different groupsizes, with max_seq_length 2048, and limit 1000.

|Model | Baseline (FP32) | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------|-----------------| ---------------------- | ---------------
|Llama 3 8B | 7.9 | 9.4 | 9.7

Please note that LM Eval reports perplexity normalized by word count instead of token count. You may see different perplexity for WikiText from other sources if they implement it differently. More details could be found [here](https://github.com/EleutherAI/lm-evaluation-harness/issues/2301).

### Performance

Llama 3 8B performance was measured on the Samsung Galaxy S22, S24, and OnePlus 12 devices. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-4-run-benchmark-on-android-phone).

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22  | 7.85 tokens/second | 8.4 tokens/second |
|Galaxy S24 | 10.91 tokens/second | 11.21 tokens/second |
|OnePlus 12 | 10.85 tokens/second | 11.02 tokens/second |

<p align="center">
      <br>
      <img src="llama_via_xnnpack.gif" width=300>
      <br>
      <em>
      Llama3.1 8B, 4bit quantized on Android phone
      </em>
</p>

[Please visit this section to try it on non-CPU backend, including CoreML, MPS, Qualcomm HTP or MediaTek](non_cpu_backends.md).

# Instructions

## Tested on

- MacOS M1/M2, Linux.
- For Llama 3 8B, your device may require at least 32GB RAM. If this is a constraint for you, please try the [smaller stories model](UTILS.md).

## Step 1: Setup
> :warning: **double check your python environment**: make sure `conda activate <VENV>` is run before all the bash and python scripts.

1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_executorch.sh`
2. Run `examples/models/llama/install_requirements.sh` to install a few dependencies.


## Step 2: Prepare model

### Option A: Download and export Llama3.2 1B/3B model.

1. Download `consolidated.00.pth`, `params.json` and `tokenizer.model` from [Llama website](https://www.llama.com/llama-downloads/) or [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B). For chat use-cases, download the instruct models.

2. Export model and generate `.pte` file.

- Use **original BF16** version, without any quantization.
```
# No quantization
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=path/to/consolidated.00.pth
LLAMA_PARAMS=path/to/params.json

python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_bf16.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}" \
```
For convenience, an [exported ExecuTorch bf16 model](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/llama3_2-1B.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-ET/blob/main/ExportRecipe_1B.ipynb).

- To use **SpinQuant**, here are two ways:
    - Download directly from [Llama website](https://www.llama.com/llama-downloads). The model weights are prequantized and can be exported to `pte` file directly.
    - Follow its [instruction](https://github.com/facebookresearch/SpinQuant/tree/main?tab=readme-ov-file#3-export-to-executorch) for exporting checkpoint to ExecuTorch and then export the SpinQuant checkpoint.

```
# SpinQuant
# Set these paths to point to the exported files
LLAMA_QUANTIZED_CHECKPOINT=path/to/spinquant/consolidated.00.pth.pth
LLAMA_PARAMS=path/to/spinquant/params.json

python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_xnnpack_spinquant.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_QUANTIZED_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}"
```
For convenience, an [exported ExecuTorch SpinQuant model](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/blob/main/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/blob/main/Export_Recipe_Llama_3_2_1B_Instruct_SpinQuant_INT4_EO8.ipynb).


- To use **QAT+LoRA**, download directly from [Llama website](https://www.llama.com/llama-downloads). The model weights are prequantized and can be exported to `pte` file directly by:

```
# QAT+LoRA
# Set these paths to point to the exported files
LLAMA_QUANTIZED_CHECKPOINT=path/to/qlora/consolidated.00.pth.pth
LLAMA_PARAMS=path/to/qlora/params.json

python -m extension.llm.export.export_llm \
    --config examples/models/llama/config/llama_xnnpack_qat.yaml \
    +base.model_class="llama3_2" \
    +base.checkpoint="${LLAMA_QUANTIZED_CHECKPOINT:?}" \
    +base.params="${LLAMA_PARAMS:?}" \
```
For convenience, an [exported ExecuTorch QAT+LoRA model](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET/blob/main/Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte) is available on Hugging Face. The export was created using [this detailed recipe notebook](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET/blob/main/Export_Recipe_Llama_3_2_1B_Instruct_QLORA_INT4_EO8.ipynb).

### Option B: Download and export Llama 3 8B instruct model

You can export and run the original Llama 3 8B instruct model.

1. Llama 3 pretrained parameters can be downloaded from [Meta's official Llama 3 repository](https://github.com/meta-llama/llama3/).

2. Export model and generate `.pte` file
```
python -m extension.llm.export.export_llm \
    --config examples/models/llama/config/llama_q8da4w.yaml \
    +base.model_class="llama3" \
    +base.checkpoint=<consolidated.00.pth.pth> \
    +base.params=<params.json>
```

Due to the larger vocabulary size of Llama 3, we recommend quantizing the embeddings with `quantization.embedding_quantize=\'4,32\'` as shown above to further reduce the model size.


If you're interested in deploying on non-CPU backends, [please refer the non-cpu-backend section](non_cpu_backends.md)

## Step 3: Run on your computer to validate

1. Build executorch with optimized CPU performance as follows. Build options available [here](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt#L59).
    ```
    cmake --preset llm -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake-out

    cmake --build cmake-out -j16 --target install --config Release
    ```
Note for Mac users: There's a known linking issue with Xcode 15.1. Refer to the section of Common Issues and Mitigations below for solutions.

2. Build llama runner.
```
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
	-DBUILD_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-Bcmake-out/examples/models/llama \
	examples/models/llama

cmake --build cmake-out/examples/models/llama -j16 --config Release
```

3. Run model. Run options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/main.cpp#L18-L40).
```
cmake-out/examples/models/llama/llama_main --model_path=<model pte file> --tokenizer_path=<tokenizer.model> --prompt=<prompt>
```

To build for CoreML backend and validate on Mac, replace `-DEXECUTORCH_BUILD_XNNPACK=ON` with `-DEXECUTORCH_BUILD_COREML=ON`

If you an error about "RE2 failed to compile pattern with lookahead:...SUPPORT_REGEX_LOOKAHEAD=ON", add "-DSUPPORT_REGEX_LOOKAHEAD=ON" when building the runner.

## Step 4: Run benchmark on Android phone

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
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
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
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON
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
adb shell "cd /data/local/tmp/llama && ./llama_main --model_path <model.pte> --tokenizer_path <tokenizer.model> --prompt \"What is the capital of France?\" --seq_len 120" --warmup=1
```
## Step 5: Build Mobile apps

### iOS

Please refer to [this tutorial](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/apple) to for full instructions on building the iOS etLLM Demo App.

### Android
Please refer to [this tutorial](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android) to for full instructions on building the Android LLAMA Demo App.

## Running with low-bit kernels

We now give instructions for quantizating and running your model with low-bit kernels.  These are still experimental, and require you do development on an Arm-based Mac, and install executorch from source with the environment variable EXECUTORCH_BUILD_KERNELS_TORCHAO=1 defined:
```
EXECUTORCH_BUILD_KERNELS_TORCHAO=1 python install_executorch.py
```

(If you'd like lowbit to use KleidiAI when available, you can instead install with `EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_KLEIDIAI=1 python install_executorch.py`.)

Also note that low-bit quantization often requires QAT (quantization-aware training) to give good quality results.

First export your model for lowbit quantization (step 2 above):

```
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=path/to/consolidated.00.pth.pth
LLAMA_PARAMS=path/to/params.json

# Set low-bit quantization parameters
QLINEAR_BITWIDTH=4 # Can be 1-8
QLINEAR_GROUP_SIZE=128 # Must be multiple of 16
QEMBEDDING_BITWIDTH=4 # Can be 1-8
QEMBEDDING_GROUP_SIZE=32 # Must be multiple of 16

python -m extension.llm.export.export_llm \
  base.model_class="llama3_2" \
  base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  base.params="${LLAMA_PARAMS:?}" \
  model.use_kv_cache=True \
  model.use_sdpa_with_kv_cache=True \
  base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' \
  export.output_name="llama3_2.pte" \
  quantization.qmode="torchao:8da${QLINEAR_BITWIDTH}w" \
  quantization.group_size=${QLINEAR_GROUP_SIZE} \
  quantization.embedding_quantize=\'torchao:${QEMBEDDING_BITWIDTH},${QEMBEDDING_GROUP_SIZE}\' \
  model.dtype_override="fp32"
```

A few notes:
- If your model shares embedding/unembedding weights (like Llama1B and Llama3B do), you can add `model.use_shared_embedding=True` to take advantage of this and reduce memory.  When this option is enabled, you can specify whether embeddings are quantized asymmetrically or not by specifying a third argument.  For example, `quantization.embedding_quantize="torchao:4,32,true"` means that the embedding is quantized to 4-bits with group_size=32 and is asymmetric (this is the default behavior if you simply use `quantization.embedding_quantize="torchao:4,32"`), whereas `quantization.embedding_quantize="torchao:4,32,false"` means that the embedding is quantized to 4-bits with group_size=32 and is symmetric.  If `model.use_shared_embedding=True` is specified, the unembedding (i.e., the final linear layer) is quantized in the same way, but also uses 8-bit dynamically quantized activations.
- To do channelwise quantization, specify group_size to 0.  This works for both linear and embedding layers.

Once the model is exported, we need to build ExecuTorch and the runner with the low-bit kernels.

The first step is to install ExecuTorch (the same as step 3.1 above):

```
cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_TORCHAO=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -Bcmake-out .
cmake --build cmake-out -j16 --config Release --target install
```

Next install the llama runner with torchao kernels enabled (similar to step 3.2 above):

```
cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-out/examples/models/llama \
    examples/models/llama
cmake --build cmake-out/examples/models/llama -j16 --config Release
```

Finally run your model (similar to step 3.3 above):

```
cmake-out/examples/models/llama/llama_main --model_path=<model pte file> --tokenizer_path=<tokenizer.model> --prompt=<prompt>
```

## Utility tools for Llama enablement

### Evaluate model accuracy

> Forewarning: Model evaluation without a GPU may take a long time, especially on larger models.

We use [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate model accuracy.

For base models, use the following example command to calculate its perplexity based on WikiText.
```
python -m examples.models.llama.eval_llama \
	-c <consolidated.00.pth.pth> \
	-p <params.json> \
	-t <tokenizer.model/bin> \
	-kv \
	-d <checkpoint dtype> \
	--max_seq_len <max sequence length> \
	--max_context_len <max context length> \
	--limit <number of samples>
```

For instruct models, use the following example command to calculate its MMLU score.
```
python -m examples.models.llama.eval_llama \
	-c <consolidated.00.pth.pth> \
	-p <params.json> \
	-t <tokenizer.model/bin> \
	-kv \
	-d <checkpoint dtype> \
	--tasks mmlu \
	--num_fewshot 5 \
	--max_seq_len <max sequence length> \
	--max_context_len <max context length>
```

See [Llama utils page](UTILS.md) page for more advanced use-cases such as fine-tuning and running smaller models for educational purposes, and quick iteration and verification.

# What is coming next?
## Quantization
- Enabling FP16 model to leverage smaller groupsize for 4-bit quantization.
- Enabling GPTQ for 4-bit groupwise quantization
- Enabling custom quantization
- Lower bit quantization
## Models
- Enabling more generative AI models and architectures.
## Performance
- Performance improvement via techniques such as speculative decoding
- Enabling LLama and other architectures via Vulkan
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
./install_executorch.sh --clean
./install_executorch.sh
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

- If you encounter issues with missing abseil-cpp or re2, try running `git submodule update --init --recursive` to pull in those submodules.
Example error:
```
CMake Error at runner/CMakeLists.txt:68 (add_subdirectory):
  The source directory

    /Users/../executorch/extension/llm/tokenizers/third-party/abseil-cpp

  does not contain a CMakeLists.txt file.


CMake Error at runner/CMakeLists.txt:72 (add_subdirectory):
  The source directory

    /Users/../executorch/extension/llm/tokenizers/third-party/re2

  does not contain a CMakeLists.txt file.
```
