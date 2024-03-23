# Summary
This is an experimental feature to convert [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) to PTE file, which can be executed directly on ExecuTorch.

## Usage:

    python executorch/extension/gguf_util/convert_main.py --gguf_file=<path_to_gguf_file> --pte_file=<output_pte_file>
