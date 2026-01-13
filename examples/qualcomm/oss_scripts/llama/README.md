# Summary

## Overview

**Video Tutorial:** [Build Along: Run LLMs Locally on Qualcomm Hardware Using ExecuTorch](https://www.youtube.com/watch?v=41PKDlGM3oU)

This file provides you the instructions to run LLM Decoder model with different parameters via Qualcomm HTP backend. We currently support the following models:
<!-- numbered list will be automatically generated -->
 1. LLAMA2 Stories 110M
 1. LLAMA3.2 1B
 1. LLAMA3.2 3B
 1. Codegen2 1B
 1. Gemma 2B
 1. Gemma3 1B
 1. GLM 1.5B
 1. Granite3.3 2B
 1. Phi4-mini-instruct
 1. QWEN2.5 0.5B / 1.5B
 1. QWEN3 0.6B / 1.7B
 1. SmolLM2 135M
 1. SmolLM3 3B


We offer the following modes to execute the model:

- KV Cache Mode: In KV Cache mode, the model takes in a single previous token and generates the next predicted token along with its KV cache. It is efficient for generating subsequent tokens after the initial prompt.

- Hybrid Mode: Hybrid mode leverages the strengths of both AR-N model and KV cache modes to optimize token generation speed. Initially, it uses AR-N model to efficiently generate the prompt's key-value (KV) cache. Then, the mode switches to KV cache mode, which excels at generating subsequent tokens.
  - AR-N model: The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use it to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor in hybrid mode.
  - Prompt processing with AR-N model: 
  <figure>
    <img src="assets/PromptProcessingWithARN.png" alt="Prompt Processing With AR-N Model">
    <figcaption>Prompt processing is done using a for-loop. An N-token block is taken, and the KV cache is updated for that block. This process is repeated until all tokens are consumed, with the last block potentially requiring padding. For flexibility, the AR-N model can handle any input length less than the maximum sequence length. For TTFT, the input length (or number of blocks) will vary depending on the actual input length, rather than always being the same.
    </figcaption>
</figure>

- Lookahead Mode: Lookahead Mode introduces [lookahead decoding](https://arxiv.org/abs/2402.02057) and uses AR-N model to process prompt to enhance token generation speed. While decoding multiple tokens in a single step is infeasible, an LLM can generate multiple guess tokens in parallel. These guess tokens may fit into future parts of the generated sequence. The lookahead decoder generates and verifies these guess tokens, integrating them into the sequence if suitable. In some cases, it can obtain more than one token in a single step. Result is lossless.

## Hardware Support

We’ve validated this flow on the **Samsung Galaxy S23**, **Samsung Galaxy S24**, **Samsung Galaxy S25**, and **OnePlus 12**.  
Support on other hardware depends on the **HTP architecture (HtpArch)** and the feature set available on that version.

### HTP Minimum Version Requirements

- **LPBQ (16a4w block-wise quantization)** requires **V69 or newer**
- **Weight sharing** between prefill and decode requires **V73 or newer**
- **16-bit activations + 16-bit weights for matmul** (e.g., 16-bit KV cache) requires **V73 or newer**

### Quantization Guidance for Older Devices

For older HTP versions, you may need to adjust the quantization strategy. Recommended starting points:

- Use **16a4w** as the baseline
- Optionally apply **SpinQuant**
- Use **16a8w selectively on some layers** to further improve accuracy (mixed-precision quantization)

### Memory Limit Errors (4 GB HTP Limit)

If you encounter errors like the following, it typically means the model’s requested memory exceeds the **4 GB per-context limit** on HTP.  
To resolve this, try **increasing the sharding number** (`num_sharding`) to reduce per-shard memory usage:

```
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0 with context size estimate 4025634048
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> context create from binary failed on contextId 1
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Fail to create context from binary with err 1002
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Size Calculation encounter error! Doing Hard reset of reserved mem to 0.
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to create context from binary with err 0x3ea
[ERROR] [Qnn ExecuTorch]: Can't create context from binary
```


## Instructions
### Note
1. For hybrid mode, the export time will be longer and can take up to 1-4 hours to complete, depending on the specific model users are exporting.
2. When exporting a hybrid mode model, memory consumption will be higher. Taking LLAMA3.2 1B as an example, please ensure the device has at least 80 GB of memory and swap space.


### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/main/backends-qualcomm) to build Qualcomm AI Engine Direct Backend.
3. Please install the llm eval dependency via [examples/models/llama/install_requirements.sh](https://github.com/pytorch/executorch/blob/main/examples/models/llama/install_requirements.sh)

### Step 2: Prepare Model

#### LLAMA2
Download and prepare stories110M model

```bash
# tokenizer.model & stories110M.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"

# tokenizer.bin:
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

#### LLAMA3.2
Follow the [instructions](https://www.llama.com/) to download models.
At the end of this step, users should have the following files ready: `consolidated.00.pth`, `params.json`, and `tokenizer.model`.


### Step3: Run default examples.
#### Note:
All example scripts below use hybrid mode, which is optimized for on-device performance. However, compiling a model in hybrid mode can consume a significant amount of memory on the host machine—sometimes up to ~100 GB. If your host machine has limited memory, it is highly recommended to switch from `--model_mode hybrid` to `--model_mode kv` and remove the `--prefill_ar_len` flag.

#### LLAMA2
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint stories110M.pt --params params.json --tokenizer_model tokenizer.model --tokenizer_bin tokenizer.bin --decoder_model stories110m --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "Once upon a time"
```

#### LLAMA3.2 1B Instruct
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### LLAMA3.2 3B Instruct
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Codegen2
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model codegen2_1b --model_mode kv --max_seq_len 1024 --prompt "def hello_world():" 
```

#### Gemma 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma-2b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```


#### Gemma3 1B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma3-1b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### GLM 1.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model glm-1_5b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Granite3.3 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model granite_3_3-2b_instruct --prompt "I would like to learn python, could you teach me with a simple example?" --run_lm_eval --task hellaswag --limit 10
```

#### Phi4-mini-instruct
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model phi_4_mini --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN2.5 0.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen2_5-0_5b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN2.5 1.5B
Default example using kv mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN3 0.6B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen3-0_6b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN3 1.7B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen3-1_7b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### SmolLM2
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smollm2_135m --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### SmolLM3
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smollm3-3b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

### KV Cache update mechanism
We use Smart Mask mechanisms for updating the key-value (KV) cache.

#### Smart Mask mechanism:
<figure>
    <img src="assets/SmartMask.png" alt="Smart Mask mechanism">
    <figcaption>The figure illustrates how key and value caches are updated during each inference step. The Smart Mask mechanism simplifies updating tokens in the cache by modifying only the new token at the designated position. This approach is useful for shared buffers, though it does require copying data in CPU memory to update the kv cache. </figcaption>
</figure>

#### Analysis KV Cache Update Mechanism for each Layer each inference
<table>
  <tr>
    <th>Mechanism</th>
    <th colspan="2" style="text-align:center;">Time Complexity</th>
    <th colspan="2" style="text-align:center;">Space Complexity</th>
  </tr>
  <tr>
    <th></th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
  </tr>
  <tr>
    <td style="text-align:center;">Smart Mask</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
  </tr>
</table>

### Additional Configs when running the script

#### Compile Only
If you would like to compile the model only, we have provided the flag `--compile_only`. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "what is 1+1" --compile_only
```

#### Pre Generated PTE
On the other hand, if you already have a pre-compiled .pte model, you can perform inference by providing the flag `--pre_gen_pte` and specifying the folder that contains the .pte model. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "what is 1+1" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

#### Lookahead Decoding Mode

You can choose the lookahead mode to enhance decoding speed. To use this mode, you need to specify the following parameters:
- `--ngram` (N-gram size): Represents the size of the n-grams used in the lookahead process.
- `--window` (window size): Determines how many future tokens the algorithm attempts to predict in each step.
- `--gcap` (Verification candidates): Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.

For more details, please refer to the paper ["Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)

```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode lookahead --prefill_ar_len 32 --max_seq_len 128 --prompt "what is 1+1" --ngram 3 --window 2 --gcap 2
```

#### Perplexity Evaluation
This script supports perplexity evaluation and is capable of assessing perplexity scores across 3 phases: prepare_pt2e(CPU FP), convert_pt2e(CPU QDQ), QNN on device.

To evaluate the perplexity across all 3 phases, users should provide the `--run_lm_eval` flag and specify the evaluation task. Please notice when this flag is provided, the `--prompt ${PROMPT}` will be ignored.

For example, using the Qwen model and 1 wikitext sample as the evaluation task, users can assess all 3 phases perplexity score in a single run by including the appropriate configuration:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "What is 1+1?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --run_lm_eval --tasks wikitext --limit 1
```

For the example script above, 1 wikitext sample is used to evaluate all 3 phases. However, there are cases where a user may want to use one sample for quantization calibration and multiple samples for perplexity evaluation. In this case, the process should be split into two runs. In the 1st run, the model is compiled using one sample. In the 2nd run, the user can provide a different configuration for QNN device execution.
Example:
```bash
# 1st run to compile with --limit 1
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "What is 1+1?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --run_lm_eval --tasks wikitext --limit 1 --compile_only
```
```bash
# 2nd run to perform QNN device execution with --limit 3
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "What is 1+1?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --run_lm_eval --tasks wikitext --limit 3 --pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN} --quant_attrs_path ${PATH_TO_ARTIFACT_IN_1ST_RUN}/kv_llama_qnn_quant_attrs.json
```

#### Tasks quantization calibration
If `--tasks ${TASK}` is not provided, the program will use `--prompt ${PROMPT}` as the dataset for quantization calibration.
Regardless of whether `--run_lm_eval` is provided, as long as `--tasks ${TASK}` is specified, the specified tasks will be used for model quantization calibration instead of the prompt.
