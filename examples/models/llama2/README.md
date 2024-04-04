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

Performance was measured on Samsung Galaxy S22, S23, S24 and One Plus 12. Measurement performance is in terms of tokens/second.

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22 | x | x |
|Galaxy S24 | x | x |
|One plus 12 | x | x |
|iPhone 15 pro | x | x |


# Instructions

## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch
2. Run `examples/models/llama2/install_requirements.sh` to install a few dependencies.

## Step 2: Prepare model

### Option A: Download and export llama2 7B model

You can export and run the original Llama2 7B model.

1. Llama2 pretrained parameters can be downloaded [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

2. TODO: Do some preparation.

3. Export model and generate `.pte` file:
    ```
    python -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w -d fp32
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
    python -m examples.models.llama2.export_llama -c stories110M.pt -p params.json
    ```
4. Create tokenizer.bin.

    Build with buck2:
    ```
    python -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
    ```

## Step 3: Run on your computer to validate

1. Build llama runner. TODO

2. Run model. Run options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/main.cpp#L13).
    Build with buck2:
    ```
    buck2 run examples/models/llama2:main -- --model_path=llama2.pte --tokenizer_path=tokenizer.bin --prompt="Once"
    ```
    Build with cmake: TODO

## Step 4: Run benchmark on Android phone

1. Build llama runner binary for Android

2. Run on Android via adb shell

## Step 5: Build iOS and/or Android apps

TODO

# What is coming next?

TODO

# Notes
This example tries to reuse the Python code, with minimal modifications to make it compatible with current ExecuTorch:
1. Since ExecuTorch does not support complex Tensor data type, use the customized functions to have rotary embedding with real numbers. Please see [GitHub issue: Support complex data type in ExecuTorch](https://github.com/pytorch/executorch/issues/886).
2. No CUDA. ExecuTorch is focused on Edge use cases where CUDA is not available on most of the edge devices.
3. No dependencies on fairscale. The ColumnParallelLinear, ParallelEmbedding and training are not needed and supported in ExecuTorch.
