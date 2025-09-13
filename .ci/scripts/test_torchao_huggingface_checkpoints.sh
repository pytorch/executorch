#!/usr/bin/env bash
set -euxo pipefail

# -------------------------
# Args / flags
# -------------------------
TEST_WITH_RUNNER=0
MODEL_NAME=""

# Parse args
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [--test_with_runner]"
  echo "Supported model_name values: qwen3_4b, phi_4_mini"
  exit 1
fi

MODEL_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test_with_runner)
      TEST_WITH_RUNNER=1
      ;;
    -h|--help)
      echo "Usage: $0 <model_name> [--test_with_runner]"
      echo "  model_name: qwen3_4b | phi_4_mini"
      echo "  --test_with_runner: build ET + run llama_main to sanity-check the export"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

MODEL_OUT=model.pte

case "$MODEL_NAME" in
  qwen3_4b)
    echo "Running Qwen3-4B export..."
    HF_MODEL_DIR=$(hf download pytorch/Qwen3-4B-INT8-INT4)
    EXPECTED_MODEL_SIZE_UPPER_BOUND=$((3 * 1024 * 1024 * 1024)) # 3GB
    $PYTHON_EXECUTABLE -m executorch.examples.models.qwen3.convert_weights \
      $HF_MODEL_DIR \
      pytorch_model_converted.bin

    $PYTHON_EXECUTABLE -m executorch.examples.models.llama.export_llama \
      --model "qwen3_4b" \
      --checkpoint pytorch_model_converted.bin \
      --params examples/models/qwen3/config/4b_config.json \
      --output_name $MODEL_OUT \
      -kv \
      --use_sdpa_with_kv_cache \
      -X \
      --xnnpack-extended-ops \
      --max_context_length 1024 \
      --max_seq_length 1024 \
      --dtype fp32 \
      --metadata '{"get_bos_id":199999, "get_eos_ids":[200020,199999]}'
    ;;

  phi_4_mini)
    echo "Running Phi-4-mini export..."
    HF_MODEL_DIR=$(hf download pytorch/Phi-4-mini-instruct-INT8-INT4)
    EXPECTED_MODEL_SIZE_UPPER_BOUND=$((3 * 1024 * 1024 * 1024)) # 3GB
    $PYTHON_EXECUTABLE -m executorch.examples.models.phi_4_mini.convert_weights \
      $HF_MODEL_DIR \
      pytorch_model_converted.bin

    $PYTHON_EXECUTABLE -m executorch.examples.models.llama.export_llama \
      --model "phi_4_mini" \
      --checkpoint pytorch_model_converted.bin \
      --params examples/models/phi_4_mini/config/config.json \
      --output_name $MODEL_OUT \
      -kv \
      --use_sdpa_with_kv_cache \
      -X \
      --xnnpack-extended-ops \
      --max_context_length 1024 \
      --max_seq_length 1024 \
      --dtype fp32 \
      --metadata '{"get_bos_id":199999, "get_eos_ids":[200020,199999]}'
    ;;

  *)
    echo "Error: unsupported model_name '$MODEL_NAME'"
    echo "Supported values: qwen3_4b, phi_4_mini"
    exit 1
    ;;
esac

# Check file size
MODEL_SIZE=$(stat --printf="%s" $MODEL_OUT 2>/dev/null || stat -f%z $MODEL_OUT)
if [[ $MODEL_SIZE -gt $EXPECTED_MODEL_SIZE_UPPER_BOUND ]]; then
  echo "Error: model size $MODEL_SIZE is greater than expected upper bound $EXPECTED_MODEL_SIZE_UPPER_BOUND"
  exit 1
fi

# Install ET with CMake
if [[ "$TEST_WITH_RUNNER" -eq 1 ]]; then
  echo "[runner] Building and testing llama_main ..."
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
        -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
        -Bcmake-out .
    cmake --build cmake-out -j16 --config Release --target install


    # Install llama runner
    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_BUILD_TYPE=Release \
        -Bcmake-out/examples/models/llama \
        examples/models/llama
    cmake --build cmake-out/examples/models/llama -j16 --config Release

    # Run the model
    ./cmake-out/examples/models/llama/llama_main --model_path=$MODEL_OUT --tokenizer_path="${HF_MODEL_DIR}/tokenizer.json" --prompt="Once upon a time,"
fi

# Clean up
rm -f pytorch_model_converted.bin "$MODEL_OUT"
