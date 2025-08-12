# Summary
For Llama enablement, please see the [Llama README page](../llama/README.md) for complete details.

This page contains Llama2 specific instructions and information.


## Enablement

We have verified running Llama 2 7B [mobile applications](#step-6-build-mobile-apps) efficiently on select devices including the iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S22 and S24, and OnePlus 12.

Since Llama 2 7B needs at least 4-bit quantization to fit even within some of the highend phones, results presented here correspond to 4-bit groupwise post-training quantized model.

## Results

### Llama2 7B
Llama 2 7B performance was measured on the Samsung Galaxy S22, S24, and OnePlus 12 devices. The performance measurement is expressed in terms of tokens per second using an [adb binary-based approach](#step-5-run-benchmark-on).

|Device  | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------| ---------------------- | ---------------
|Galaxy S22  | 8.15 tokens/second | 8.3 tokens/second |
|Galaxy S24 | 10.66 tokens/second | 11.26 tokens/second |
|OnePlus 12 | 11.55 tokens/second | 11.6 tokens/second |

Below are the results for two different groupsizes, with max_seq_length 2048, and limit 1000, based on WikiText perplexity using [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness).

|Model | Baseline (FP32) | Groupwise 4-bit (128) | Groupwise 4-bit (256)
|--------|-----------------| ---------------------- | ---------------
|Llama 2 7B | 9.2 | 10.2 | 10.7

## Prepare model

You can export and run the original Llama 2 7B model.

1. Llama 2 pretrained parameters can be downloaded from [Meta's official website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) or from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b).

2. Edit `params.json` file. Replace `"vocab_size": -1` with `"vocab_size": 32000`. This is a short-term workaround.

3. Export model and generate `.pte` file:
    ```
    python -m extension.llm.export.export_llm base.checkpoint=<checkpoint.pth> base.params=<params.json> model.use_kv_cache=True model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True quantization.qmode="8da4w" quantization.group_size=128 model.dtype_override="fp32"
    ```
4. Create tokenizer.bin.
    ```
    python -m pytorch_tokenizers.tools.llama2c.convert -t <tokenizer.model> -o tokenizer.bin
    ```

    Pass the converted `tokenizer.bin` file instead of `tokenizer.model` for subsequent steps.


# Run

Running will be the same [by following this step](../llama/README.md#step-4-run-on-your-computer-to-validate).
