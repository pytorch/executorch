# Run Llama 3 3B Instruct on Android (with Qualcomm AI Engine Direct Backend)

This tutorial demonstrates how to export and run the Llama 3 3B Instruct model on a Qualcomm device using the Qualcomm AI Engine Direct Backend via ExecuTorch.
We use a static Llama [implementation](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/model/static_llama.py) to optimize performance and memory usage during on-device inference.

## Prerequisites

- Set up your ExecuTorch repo and environment if you haven’t done so by following [the Setting up ExecuTorch](../getting-started-setup.rst) to set up the repo and dev environment.
- Read [the Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend page](../backends-qualcomm.md) to understand how to export and run a model with Qualcomm AI Engine Direct Backend on Qualcomm device.
- Follow [the README for executorch llama](https://github.com/pytorch/executorch/tree/main/examples/models/llama) to know how to run a llama model on mobile via ExecuTorch.
- A Qualcomm device with 16GB RAM
  - We are continuing to optimize our memory usage to ensure compatibility with lower memory devices.
- The version of [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) is 2.28.0 or above.

## Instructions

### Step 1: Prepare the checkpoint and tokenizer of the model.
1. For Llama 3 tokenizer and checkpoint, please refer to [instructions](https://www.llama.com/models/llama-3) for further instructions on how to download `tokenizer.model`, `consolidated.00.pth` and `params.json`.

### Step 2: Export to ExecuTorch with Qualcomm AI Engine Direct Backend
Deploying large language models like Llama 3 on-device presents the following challenges:

1. The model size is too large to fit in device memory for inference.
2. High model loading and inference time.
3. Difficulty in quantization.

To address these, we apply the following optimizations:

1. Quantization: Use `QuantDtype.use_16a4w_block` for post-training quantization to reduce model size and memory usage

2. Mixed Precision Quantization: compresses KV cache tensors to 8-bit and applies `QuantDtype.use_16a8w` to the LM head.

3. SeqMSE Quantization: optimizes the parameter encodings of each layer of a model individually to minimize the difference between the layer’s original and quantized outputs. SeqMSE uses a search-based approach with `seq_mse_candidates` = 1000. (Implementation details: [SeqMSE pass](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/_passes/seq_mse.py))

4. Model Sharding: Set `num_sharding` = 4 to shard the model into sub-parts. This helps reduce memory pressure and improve performance during on-device inference.

5. Graph Transformations: Convert operations into accelerator-friendly formats for better runtime performance.

You can find the full optimization configuration in this [file](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/__init__.py), as shown below:

``` python
@register_llm_model("llama3_2-1b_instruct")
@dataclass(init=False, frozen=True)
class Llama3_2_1B_Instruct(LLMModelConfig):
    repo_id = None
    params_path = None
    convert_weights = None
    transform_weight = True
    # The Llama3_2 enabled should be instruct, however, Llama's tokenizer does not provide utility to apply chat template.
    instruct_model = False
    
    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 32
    masked_softmax = False
    seq_mse_candidates = 1000
    r1 = False
    r2 = False
    r3 = False
    quantization_config_down_proj_16a8w = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8, act_observer=MinMaxObserver
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(
            annotate_down_proj, quantization_config=quantization_config_down_proj_16a8w
        ),
    )
```


To export with the Qualcomm AI Engine Direct Backend, ensure the following:

1. The host machine has more than 64GB of memory (RAM + swap space).
2. The entire process takes a few hours.

```bash
# export llama
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode kv --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1 --compile_only
```
Note: end-to-end [instructions](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md)

### Step 3: Invoke the Runtime on an Android smartphone with Qualcomm SoCs
**3.1 Connect your android phone**

**3.2 Make sure the following artifact is present before running the model.**
-- artifact/
   └── llama_qnn.pte

**3.3 Run model**
```bash
# Run llama
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode kv --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1 --pre_gen_pte ${PATH_TO_ARTIFACT}
```

## What is coming?
- Performance improvements
- Reduce the memory pressure during inference to support 12GB Qualcomm devices
- Broader LLM Support via [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch?tab=readme-ov-file#llms-large-language-models)

  - Already supported models (e.g.): Llama2, Llama3, Gemma, Qwen, Phi-4, SmolLM. For usage examples, please refer to [README](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md)

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#qcom_aisw` tag