# Build the runtime

Do this after modifying runtime code (cpp)
```sh
cmake --workflow llm-debug
pushd examples/models/llama
cmake --workflow --preset llama-debug
popd
```

# Export model
Take a look at examples/models/llama/README.md

Check point is in ~/executorch/

Make sure you are in conda executorch env

# No quantization
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=path/to/consolidated.00.pth
LLAMA_PARAMS=path/to/params.json

python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_bf16.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="consolidated.00.pth" \
  +base.params="params.json"
```

# Run

Please also take a look at examples/models/llama/runner to make sure it can emit many tokens, exceeding context size.

Please check whether the output makes sense or not
```
cmake-out/examples/models/llama/llama_main --model_path=<model pte file> --tokenizer_path=<tokenizer.model> --prompt=<prompt>
```
