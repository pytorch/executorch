## Tutorial to run [eval_decoder_model_qnn.py](./eval_decoder_model_qnn.py)
This script, [`eval_decoder_model_qnn.py`](./eval_decoder_model_qnn.py), is designed to evaluate large language models (LLMs) from transformers that have been compiled into ExecuTorch Portable Executable (PTE) format for execution on Qualcomm devices. It leverages the `lm-evaluation-harness` library to perform various NLP evaluation tasks.

> ⚠️ **Important:** Note that this script runs PTE files generated specifically for Hugging Face Transformers, such as [qwen2_5.py](../qwen2_5/qwen2_5.py), rather than [the static LLaMA version](../llama/llama.py).

### Features:

*   Evaluates ExecuTorch PTE models on Qualcomm devices (requires ADB setup and QNN SDK).
*   Integrates with `lm-evaluation-harness` for standardized LLM evaluation tasks.

### Prerequisites

Before running this script, ensure you have the following:

1. **Setup ExecuTorch** Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. **Setup QNN ExecuTorch** Follow the [tutorial](https://pytorch.org/executorch/main/backends-qualcomm) to build Qualcomm AI Engine Direct Backend.
3.  **`lm-evaluation-harness`:** Install the `lm-evaluation-harness` library.
4.  **PTE Model:** A pre-exported ExecuTorch PTE model of your decoder LLM.
5.  **Tokenizer:** A tokenizer in a format supported by `pytorch-tokenizers` (e.g., SentencePiece `.json` or `.model`, Tiktoken).

### How to Use

The script evaluates the model by running the PTE file on a connected Qualcomm device.

#### Command Line Arguments:

*   `-a`, `--artifact`: (Required) Path for storing generated artifacts by this example.
*   `--tokenizer_path`: (Required) Path to your tokenizer file (e.g., `tokenizer.model` or `tokenizer.json`).
*   `--pte`: (Required) Path to the ExecuTorch Portable Executable (`.pte`) model file.
*   `--logits_quant_attr_path`: (Optional) Path to a JSON file containing quantization attributes. This is needed if your PTE model uses tag quant I/O for logits and requires de-quantization before evaluation.
*   `--max_seq_len`: (Optional, default: 128) Maximum sequence length the model can process.
*   `--tasks`: (Optional, default: `["wikitext"]`) A list of `lm-evaluation-harness` tasks to evaluate. You can specify multiple tasks separated by spaces (e.g., `--tasks wikitext piqa`).
*   `--limit`: (Optional) Number of samples to evaluate per task. If not set, all samples will be evaluated.
*   `--num_fewshot`: (Optional) Number of examples to use in few-shot context for evaluation.
*   `--model`: (Required for QNN execution) The SoC model name (e.g., `SM8550`, `SM8650`).
*   `--device`: (Required for QNN execution) The ADB device ID.
*   `--host`: (Required for QNN execution) The ADB host ID (usually `localhost`).
*   `--build_folder`: (Optional, default: `build-android`) The build folder for ExecuTorch artifacts, relative to the current directory.

#### Example Usage:

```bash
python examples/qualcomm/oss_scripts/llm_utils/eval_decoder_model_qnn.py \
    --artifact ./eval_output \
    --tokenizer_path /path/to/your/tokenizer.model \
    --pte /path/to/your/model.pte \
    --model SM8550 \
    --device YOUR_DEVICE_ID \
    --host localhost \
    --tasks wikitext \
    --limit 1 \
    --max_seq_len 512
```

Replace `/path/to/your/tokenizer.model`, `/path/to/your/model.pte`, and `YOUR_DEVICE_ID` with your actual paths and device ID.

If your model's logits are quantized and require de-quantization:

```bash
python examples/qualcomm/oss_scripts/llm_utils/eval_decoder_model_qnn.py \
    --artifact ./eval_output \
    --tokenizer_path /path/to/your/tokenizer.model \
    --pte /path/to/your/model.pte \
    --logits_quant_attr_path /path/to/your/logits_quant_attrs.json \
    --model SM8550 \
    --device YOUR_DEVICE_ID \
    --host localhost \
    --tasks wikitext \
    --limit 1 \
    --max_seq_len 512
```

### Output

The script will print the evaluation results for each specified task to the console, similar to the `lm-evaluation-harness` output format. For example:

```
wikitext: {'word_perplexity': ..., 'byte_perplexity': ..., 'bits_per_byte': ...}
