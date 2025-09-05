#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"


llama_artifacts="260k_stories"
PTE_ARTIFACT="examples/qualcomm/oss_scripts/llama/artifacts"

mkdir ${llama_artifacts}
# Download stories260K.pt and tokenizer from Github
curl -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt" --output ${llama_artifacts}/stories260K.pt
curl -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model" --output ${llama_artifacts}/tokenizer.model

$PYTHON_EXECUTABLE -m pytorch_tokenizers.tools.llama2c.convert -t ${llama_artifacts}/tokenizer.model -o ${llama_artifacts}/tokenizer.bin
# Create params.json file
touch ${llama_artifacts}/params.json
echo '{"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4, "vocab_size": 512, "multiple_of": 4, "max_seq_len": 512}' > ${llama_artifacts}/params.json

# Checks e2e accuracy
expected=$($PYTHON_EXECUTABLE backends/qualcomm/tests/test_qnn_delegate.py -k TestExampleLLMScript.test_llama_stories_260k --model SM8650 --build_folder build-x86/ --executorch_root . --artifact_dir . --llama_artifacts $llama_artifacts --enable_x86_64 | grep "Model CI result:")
exit_code1=$?

# Checks accuracy with precompiled
output=$($PYTHON_EXECUTABLE backends/qualcomm/tests/test_qnn_delegate.py -k TestExampleLLMScript.test_llama_stories_260k --model SM8650 --build_folder build-x86/ --executorch_root . --artifact_dir $PTE_ARTIFACT --llama_artifacts $llama_artifacts --enable_x86_64 --pre_gen_pte $PTE_ARTIFACT | grep "Model CI result:")
exit_code2=$?

if [[ "$output" == "$expected" ]]; then
  echo "[BACKWARD COMPATIBILITY CHECK] Output matches expected result."
else
  echo "[BACKWARD COMPATIBILITY CHECK] Output mismatch!"
  echo "[BACKWARD COMPATIBILITY CHECK] Expected: $expected"
  echo "[BACKWARD COMPATIBILITY CHECK] Actual:   $output"
  exit 1
fi

# Check the exit codes and print messages
if [ $exit_code1 -ne 0 ]; then
    echo "Static Llama compile only test failed. $exit_code1."
fi

if [ $exit_code2 -ne 0 ]; then
    echo "Static Llama execute precompiled test failed. $exit_code2."
fi

# Return failure if either program failed
if [ $exit_code1 -ne 0 ] || [ $exit_code2 -ne 0  ]; then
    exit 1
else
    exit 0
fi
