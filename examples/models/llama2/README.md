# Summary
This example demonstrates how to Export a [Llama 2](https://ai.meta.com/llama/) model in ExecuTorch such that it can be used in a mobile environment.
For Llama2, please refer to [the llama's github page](https://github.com/facebookresearch/llama) for details.
Pretrained parameters are not included in this repo. Users are suggested to download them through [the llama's download page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

# What is Llama 2?
Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.

Llama models have shown to perform well on a variety of natural language processing tasks, including language translation, question answering, and text summarization and are also capable of generating human-like text, making Llama models a useful tool for creative writing and other applications where natural language generation is important.

Overall, Llama models are powerful and versatile language models that can be used for a wide range of natural language processing tasks. The model’s ability to generate coherent and contextually relevant text makes it particularly useful for applications such as chatbots, virtual assistants, and language translation.

Please note that the models are subject to the [acceptable use policy](https://github.com/facebookresearch/llama/blob/main/USE_POLICY.md) and the provided [responsible use guide](https://ai.meta.com/static-resource/responsible-use-guide/).

# Notes
1. This example is to show the feasibility of exporting a Llama2 model in ExecuTorch. There is no guarantee for performance.
2. The provided checkpoint, demo_rand_params.pth is a dummy checkpoint with random parameters. It does not provide meaningful results. It's only for the purpose of demonstration and fast iterations.  Use the options `--checkpoint <checkpoint>` and `--params <params>` for custom checkpoints.


# Limitations
This example tries to reuse the Python code, with modifications to make it compatible with current ExecuTorch:
1. Since ExecuTorch does not support complex Tensor data type, use the customized functions to have rotary embedding with real numbers. Please see [GitHub issue: Support complex data type in ExecuTorch](https://github.com/pytorch/executorch/issues/886).
2. No KV cache. The current cache implementation in the original Llama2 repo is not supported by ExecuTorch, because ExecuTorch runtime assumes model data attributes being static. Please see [GitHub issue: Add support of mutable buffers in ExecuTorch](https://github.com/pytorch/executorch/issues/897).
3. No CUDA. ExecuTorch is focused on Edge use cases where CUDA is not available on most of the edge devices.
4. No dependencies on fairscale. The ColumnParallelLinear, ParallelEmbedding and training are not needed and supported in ExecuTorch.


# Instructions:
### Setup
1. Follow the [tutorial](https://pytorch.org/executorch/stable/getting-started-setup) to set up ExecuTorch
2. `cd examples/third-party/llama`
3. `pip install -e .`
4. Go back to `executorch` root, run `bash examples/models/llama2/install_requirements.sh`.

### Export llama2 models
2. From `executorch` root, run `python3 -m examples.models.llama2.export_llama`. The exported program, llama2.pte would be saved in current directory using the dummy checkpoint.
3. Llama2 pretrained parameters can be downloaded [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and run with `python3 -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json>`.

### Export and run stories110M model

1. Download `stories110M.pt` and `tokenizer.model` from Github.
    ```
    wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
    wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
    ```
2. Create params file.
    ```
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
    ```
3. Export model. Export options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/export_llama_lib.py#L161).
    ```
    python3 -m examples.models.llama2.export_llama -c stories110M.pt -p params.json
    ```
4. Create tokenizer.bin.

    Build with buck2:
    ```
    buck2 run examples/models/llama2/tokenizer:tokenizer_py -- -t tokenizer.model -o tokenizer.bin
    ```
    Build with cmake: todo

5. Run model. Run options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/main.cpp#L13).
    Build with buck2:
    ```
    buck2 run examples/models/llama2:main -- --model_path=llama2.pte --tokenizer_path=tokenizer.bin --prompt="Once"
    ```
    Build with cmake: todo

See test script [here](https://github.com/pytorch/executorch/blob/main/.ci/scripts/test_llama.sh).
