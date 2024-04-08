# Summary
This example demonstrates how to run a [Llama 2](https://ai.meta.com/llama/) 7B model on mobile via ExecuTorch. We use XNNPACK to accelerate the performance and 4-bit groupwise PTQ quantization to fit the model on a phone.


For Llama2, please refer to [the llama's github page](https://github.com/facebookresearch/llama) for details.
Pretrained parameters are not included in this repo. Users are suggested to download them through [the llama's download page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

# What is Llama 2?
Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.

Llama models have shown to perform well on a variety of natural language processing tasks, including language translation, question answering, and text summarization and are also capable of generating human-like text, making Llama models a useful tool for creative writing and other applications where natural language generation is important.

Overall, Llama models are powerful and versatile language models that can be used for a wide range of natural language processing tasks. The model’s ability to generate coherent and contextually relevant text makes it particularly useful for applications such as chatbots, virtual assistants, and language translation.

Please note that the models are subject to the [acceptable use policy](https://github.com/facebookresearch/llama/blob/main/USE_POLICY.md) and the provided [responsible use guide](https://ai.meta.com/static-resource/responsible-use-guide/).


# Results

Since 7B Llama2 model needs at least 4-bit quantization to fit even within some of the highend phones, results presented here correspond to 4-bit groupwise post-training quantized model.

## Quantization:
We employed 4-bit groupwise per token dynamic quantization of all the linear layers of the model. Dynamic quantization refers to quantizating activations dynamically, such that quantization parameters for activations are calculated, from min/max range, at runtime. Here we quantized activations with 8bits (signed integer). Furthermore, weights are statically quantized. In our case weights were per-channel groupwise quantized with 4bit signed integer. For more information refer to this [page](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html).

We evaluated WikiText perplexity using [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness). Below are the results for two different groupsizes.

|Llama 2 | Baseline (FP32) | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------|-----------------| ---------------------- | ---------------
|Wikitext Perplexity | 9.16 | 10.2 | 10.7

Note that groupsize less than 128 was not enabled, since such model were still too large. This is because our current efforts have focused on enabling FP32 and support for FP16 is under way. What this implies for model size is that 1) embedding table is in FP32 and 2) quantized weights scales are FP32.

## Performance

Performance was measured on Samsung Galaxy S22, S24, One Plus 12 and iPhone 15 max Pro. Measurement performance is in terms of tokens/second.

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22*  | 8.15 tokens/second | 8.3 tokens/second |
|Galaxy S24* | 10.66 tokens/second | 11.26 tokens/second |
|One plus 12* | 11.55 tokens/second | 11.6 tokens/second |
|Galaxy S22** | 5.5 tokens/second | 5.9 tokens/second |
|iPhone 15 pro** | ~6 tokens/second | ~6 tokens/second |

*: Measured via adb binary based [workflow](#step-5-run-benchmark-on)

**: Measured via app based [workflow](#step-6-build-mobile-apps)

# Instructions

## Tested on

- MacOS M1/M2, Linux.
- For Llama7b, your device may require at least 32GB RAM. If this is a constraint for you, please try the smaller stories model.

## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch
2. Run `examples/models/llama2/install_requirements.sh` to install a few dependencies.

## Step 2: Prepare model

### Option A: Download and export llama2 7B model

You can export and run the original Llama2 7B model.

1. Llama2 pretrained parameters can be downloaded from [Meta's official website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) or from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b).

2. Edit `params.json` file. Replace `"vocab_size": -1` with `"vocab_size": 32000`. This is a short-term workaround.

3. Export model and generate `.pte` file:
    ```
    python -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w --group_size 128 -d fp32
    ```
4. Create tokenizer.bin.

    ```
    python -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
    ```

### Option B: Download and export stories110M model

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
    python -m examples.models.llama2.export_llama -c stories110M.pt -p params.json -X
    ```
4. Create tokenizer.bin.

    ```
    python -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
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

Using the same arguments from above
```
python -m examples.models.llama2.eval_llama -c <checkpoint.pth> -p <params.json> -t <tokenizer.model> -d fp32 --max_seq_len <max sequence length> --limit <number of samples>
```

The Wikitext results generated above used: `{max_seq_len: 2048, limit: 1000}`

## Step 4: Run on your computer to validate

1. Build executorch with optimized CPU performance as follows. Build options available [here](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt#L59).
    ```
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_OPTIMIZED=ON \
        -Bcmake-out .

    cmake --build cmake-out -j16 --target install --config Release
    ```

2. Build llama runner.
    ```
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_OPTIMIZED=ON \
        -Bcmake-out/examples/models/llama2 \
        examples/models/llama2

    cmake --build cmake-out/examples/models/llama2 -j16 --config Release
    ```

3. Run model. Run options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/main.cpp#L18-L40).
    ```
    cmake-out/examples/models/llama2/llama_main --model_path=<model pte file> --tokenizer_path=<tokenizer.bin> --prompt=<prompt>
    ```

## Step 5: Run benchmark on Android phone

**1. Build llama runner binary for Android**

*Pre-requisite*: Android NDK (tested with r26c) which can be downloaded from [here](https://developer.android.com/ndk/downloads). Note that the mac binary can be unpackaged and you can locate NDK folder from it.

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
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_OPTIMIZED=ON \
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
    -DEXECUTORCH_BUILD_OPTIMIZED=ON \
    -Bcmake-out-android/examples/models/llama2 \
    examples/models/llama2

cmake --build cmake-out-android/examples/models/llama2 -j16 --config Release
```

**2. Run on Android via adb shell**

*Pre-requisite*: Make sure you enable USB debugging via developer options on your phone

**2.1 Connect your android phone**

**2.2 Upload model, tokenizer and llama runner binary to phone**
```
adb shell mkdir -p /data/local/tmp/llama
adb push <model.pte> /data/local/tmp/llama/
adb push <tokenizer.bin> /data/local/tmp/llama/
adb push cmake-out-android/examples/models/llama2/llama_main /data/local/tmp/llama/
```

**2.3 Run model**
```
adb shell "cd /data/local/tmp/llama && ./llama_main --model_path <model.pte> --tokenizer_path <tokenizer.bin> --prompt "Once upon a time" --seq_len 120
```
## Step 6: Build Mobile apps

### iOS

Please refer to [this tutorial](https://pytorch.org/executorch/main/llm/llama-demo-ios.html) to for full instructions on building the iOS LLAMA Demo App.

### Android
Please refer to [this tutorial](https://pytorch.org/executorch/main/llm/llama-demo-android.html) to for full instructions on building the Android LLAMA Demo App.

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


# Clean
To clean your build:
```
git clean -xfd
pip uninstall executorch
./install_requirements.sh <options>

rm -rf cmake-out
```
