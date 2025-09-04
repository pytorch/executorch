# Artifacts folder for LLaMA backward compatibility validation
This folder contains the stories260K(a smaller LLaMA variant) .pte artifact for backward compatibility (BC) validation in CI pipelines.

Model source: [karpathy/tinyllamas/stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)

## Purpose
The .pte files stored here serve as reference pte to ensure that changes to the ExecuTorch do not introduce backward-incompatible changes. 

These files are used in CI to:
1. Compile story llama with the previous (n-1) commit.
2. Run and validate with the current (n) commit.

We use the stories260K model because it is a minimal LLaMA variant, making it ideal for efficient validation in CI pipelines.

## File Structure
- stories260k_hybrid_llama_qnn.pte: precompiled story llama used for backward compatibility validation.
## Updating Artifacts
To update the .pte file, follow these steps:

1. Checkout the latest commit before all your changes.

2. Download and prepare stories260K model

```bash
# tokenizer.model & stories260K.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt"
wget -O tokenizer.model "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model"

# tokenizer.bin:
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4, "vocab_size": 512, "multiple_of": 4, "max_seq_len": 512}' > params.json
```

3. Run the following command to regenerate and update .pte file: 

``` bash
# Checks accuracy with weight sharing disabled since x86 does not support weight sharing.
python backends/qualcomm/tests/test_qnn_delegate.py -k TestExampleLLMScript.test_llama_stories_260k --model SM8650 --build_folder build-x86/ --executorch_root . --artifact_dir ./examples/qualcomm/oss_scripts/llama/artifacts --llama_artifacts . --enable_x86_64 --compile_only

```
4. Commit the hybrid_llama_qnn.pte file to the repository.

5. Update this README if necessary then commit your changes.

Note: The .pte file is large (~2MB). In the future, we may host it on Hugging Face and download it during CI to reduce repository size.
