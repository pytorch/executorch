# Summary

## Overview
This file provides you the instructions to run LLAMA model with different parameters via Qualcomm HTP backend. We currently support the following models:
 1. LLAMA2 Stories 110M
 2. LLAMA3.2 1B
 3. LLAMA3.2 3B (WIP)

We offer the following modes to execute the model:

Prefill Mode: This is also known as batch prefill mode, where the model takes in a list of tokens as input and generates the next token along with the key-value (KV) cache for all tokens. This mode is efficient for encoding the user's prompt.

KV Cache Mode: In KV Cache mode, the model takes in a single previous token and generates the next predicted token along with its KV cache. It is efficient for generating subsequent tokens after the initial prompt.

Hybrid Mode: Hybrid mode leverages the strengths of both batch prefill and KV cache modes to optimize token generation speed. Initially, it uses prefill mode to efficiently generate the prompt's key-value (KV) cache. Then, the mode switches to KV cache mode, which excels at generating subsequent tokens.


## Instructions
### Note
1. For hybrid mode, the export time will be longer and can take up to 1-4 hours to complete, depending on the specific model users are exporting.
2. When exporting a hybrid mode model, memory consumption will be higher. Taking LLAMA3.2 1B as an example, please ensure the device has at least 80 GB of memory and swap space.


### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html) to build Qualcomm AI Engine Direct Backend.

### Step 2: Prepare Model

#### LLAMA2
Download and prepare stories110M model

```bash
# tokenizer.model & stories110M.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"

# tokenizer.bin:
python -m extension.llm.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

#### LLAMA3.2
Follow the [instructions](https://www.llama.com/) to download models.
At the end of this step, users should have the following files ready: `consolidated.00.pth`, `params.json`, and `tokenizer.model`.


### Step3: Run default examples using hybrid mode.
#### LLAMA2
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint stories110M.pt --params params.json --tokenizer_model tokenizer.model --tokenizer_bin tokenizer.bin --llama_model stories110m --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --prompt "Once upon a time"
```

#### LLAMA3.2
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --llama_model llama3_2 --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --prompt "what is 1+1"
```

### KV Cache update mechanism
We have two distinct mechanisms for updating the key-value (KV) cache, which can be selected at runtime. Shift Pointer and Smart Mask.

#### Shift Pointer mechanism

<figure>
    <img src="./assets/ShiftPointer.png" alt="Shift Pointer mechanism"> <figcaption>
    The figure illustrates the process of updating the key and value caches during each inference step. In key cache update process, we initially allocate memory for each layer with <code>num_head</code> size of <code>(head_dim + 1) * (seq_len - 1)</code>. After a single inference, the new key cache is copied from the key output pointer <code>k_out</code> and appended to the key cache. Subsequently, the buffer start pointer of the key cache <code>k_in</code> moves to the next token, making the previous position of the buffer start pointer unused. This process is repeated for each subsequent inference step.
    For the value cache update process, we first allocate a contiguous memory of size <code>(num_head + 1) * head_dim * (seq_len - 1)</code> for each layer, with the last head reserved for I/O shifting, After the first inference, the cache is updated by simply shifting the pointers of all heads to the next token position, making only the previous <code>head_dim * 1</code> section of the buffer start pointer <code>v_in</code> of the first head unused. This process is repeated for each subsequent inference step.</figcaption>
</figure>

#### Smart Mask mechanism:
<figure>
    <img src="./assets/SmartMask.png" alt="Smart Mask mechanism">
    <figcaption>The Smart Mask mechanism streamlines the process of updating tokens in the cache. Unlike the Shift Pointer mechanism, which requires moving the buffer start pointer <code>k_in</code>/<code>v_in</code> of the cache, the Smart Mask mechanism updates only the new token at the specified position. This approach eliminates the need to adjust the buffer start pointer. This mechanism is beneficial for shared buffers but requires CPU memory copying. </figcaption>
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
    <td style="text-align:center;">Shift Pointer</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">1</td>
    <td style="text-align:center;">num_head * (head_dim + 1) * seq_len</td>
    <td style="text-align:center;">(num_head + 1) * head_dim * (seq_len - 1)</td>
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
If you would like to compile the model only, we have provided the flag `--compile_only`. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --llama_model llama3_2 --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --prompt "what is 1+1" --compile_only
```

On the other hand, if you already have a pre-compiled .pte model, you can perform inference by providing the flag `--pre_gen_pte` and specifying the folder that contains the .pte model. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --llama_model llama3_2 --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --prompt "what is 1+1" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

You can select the KV Cache update mechanism at runtime by setting the `KV_UPDATER` variable to either "shift_pointer" or "smart_mask". By default, it is set to "smart_mask".
`KV_UPDATER` = "shift_pointer"
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --ptq 16a4w --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --llama_model llama3_2 --model_mode hybrid --prefill_seq_len 32 --kv_seq_len 128 --prompt "what is 1+1" --kv_updator ${KV_UPDATER}
```
