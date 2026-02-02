#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test CUDA/Metal model end-to-end, need to run .ci/scripts/export_model_artifact.sh first

show_help() {
  cat << EOF
Usage: test_model_e2e.sh <device> <hf_model> <quant_name> [model_dir]

Build and run end-to-end tests for CUDA/Metal models.

Arguments:
  device      cuda or metal (required)

  hf_model    HuggingFace model ID (required)
              Supported models:
                - mistralai/Voxtral-Mini-3B-2507
                - openai/whisper series (whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo})
                - google/gemma-3-4b-it
                - nvidia/parakeet-tdt

  quant_name  Quantization type (required)
              Options:
                - non-quantized
                - quantized-int4-tile-packed
                - quantized-int4-weight-only

  model_dir   Directory containing model artifacts (optional, default: current directory)
              Expected files: model.pte, aoti_cuda_blob.ptd (CUDA only)
              Tokenizers and test files will be downloaded to this directory

Examples:
  test_model_e2e.sh metal "openai/whisper-small" "non-quantized"
  test_model_e2e.sh cuda "mistralai/Voxtral-Mini-3B-2507" "quantized-int4-tile-packed" "./model_output"
  test_model_e2e.sh cuda "nvidia/parakeet-tdt" "non-quantized" "./model_output"
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  show_help
  exit 0
fi

if [ -z "${1:-}" ]; then
  echo "Error: hf_model argument is required"
  echo "Run with -h or --help for usage information"
  exit 1
fi

if [ -z "${2:-}" ]; then
  echo "Error: quant_name argument is required"
  echo "Run with -h or --help for usage information"
  exit 1
fi

set -eux

DEVICE="$1"
HF_MODEL="$2"
QUANT_NAME="$3"
# Download tokenizers, audio, and image files to this directory
MODEL_DIR="${4:-.}"

echo "Testing model: $HF_MODEL (quantization: $QUANT_NAME)"

# Make sure model.pte exists
if [ ! -f "$MODEL_DIR/model.pte" ]; then
  echo "Error: model.pte not found in $MODEL_DIR"
  exit 1
fi
# For CUDA, also check for aoti_cuda_blob.ptd (Metal embeds data in .pte)
if [ "$DEVICE" = "cuda" ] && [ ! -f "$MODEL_DIR/aoti_cuda_blob.ptd" ]; then
  echo "Error: aoti_cuda_blob.ptd not found in $MODEL_DIR"
  exit 1
fi
# Locate EXECUTORCH_ROOT from the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

pushd "$EXECUTORCH_ROOT"

# Determine model configuration based on HF model ID
case "$HF_MODEL" in
  mistralai/Voxtral-Mini-3B-2507)
    MODEL_NAME="voxtral"
    RUNNER_TARGET="voxtral_runner"
    RUNNER_PATH="voxtral"
    EXPECTED_OUTPUT="existence"
    PREPROCESSOR="voxtral_preprocessor.pte"
    TOKENIZER_URL="https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/resolve/main" # @lint-ignore
    TOKENIZER_FILE="tekken.json"
    AUDIO_URL="https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/testaudio/16000/test01_20s.wav"
    AUDIO_FILE="poem.wav"
    IMAGE_PATH=""
    ;;
  openai/whisper-*)
    MODEL_NAME="${HF_MODEL#openai/}"
    RUNNER_TARGET="whisper_runner"
    RUNNER_PATH="whisper"
    EXPECTED_OUTPUT="Mr. Quilter is the apostle of the middle classes"
    PREPROCESSOR="whisper_preprocessor.pte"
    TOKENIZER_URL="https://huggingface.co/${HF_MODEL}/resolve/main" # @lint-ignore
    TOKENIZER_FILE=""
    AUDIO_URL=""
    AUDIO_FILE="output.wav"
    IMAGE_PATH=""
    ;;
  google/gemma-3-4b-it)
    MODEL_NAME="gemma3"
    RUNNER_TARGET="gemma3_e2e_runner"
    RUNNER_PATH="gemma3"
    EXPECTED_OUTPUT="chip"
    PREPROCESSOR=""
    TOKENIZER_URL="https://huggingface.co/unsloth/gemma-3-4b-it/resolve/main" # @lint-ignore
    TOKENIZER_FILE=""
    AUDIO_URL=""
    AUDIO_FILE=""
    IMAGE_PATH="docs/source/_static/img/et-logo.png"
    ;;
  nvidia/parakeet-tdt)
    MODEL_NAME="parakeet"
    RUNNER_TARGET="parakeet_runner"
    RUNNER_PATH="parakeet"
    EXPECTED_OUTPUT="Phoebe"
    PREPROCESSOR=""
    TOKENIZER_URL=""
    TOKENIZER_FILE="tokenizer.model"
    AUDIO_URL="https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
    AUDIO_FILE="test_audio.wav"
    IMAGE_PATH=""
    ;;
  *)
    echo "Error: Unsupported model '$HF_MODEL'"
    echo "Supported models: mistralai/Voxtral-Mini-3B-2507, openai/whisper series (whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo}), google/gemma-3-4b-it, nvidia/parakeet-tdt"
    exit 1
    ;;
esac

echo "::group::Setup ExecuTorch Requirements"
./install_requirements.sh
pip list
echo "::endgroup::"

echo "::group::Prepare $MODEL_NAME Artifacts"


# Download tokenizer files (skip for parakeet which exports tokenizer with model)
if [ "$MODEL_NAME" != "parakeet" ]; then
  if [ "$TOKENIZER_FILE" != "" ]; then
    curl -L $TOKENIZER_URL/$TOKENIZER_FILE -o $MODEL_DIR/$TOKENIZER_FILE
  else
    curl -L $TOKENIZER_URL/tokenizer.json -o $MODEL_DIR/tokenizer.json
    curl -L $TOKENIZER_URL/tokenizer_config.json -o $MODEL_DIR/tokenizer_config.json
    curl -L $TOKENIZER_URL/special_tokens_map.json -o $MODEL_DIR/special_tokens_map.json
  fi
fi

# Download test files
if [ "$AUDIO_URL" != "" ]; then
  curl -L $AUDIO_URL -o ${MODEL_DIR}/$AUDIO_FILE
elif [[ "$MODEL_NAME" == *whisper* ]]; then
  conda install -y -c conda-forge "ffmpeg<8"
  pip install datasets soundfile
  pip install torchcodec==0.10.0.dev20251211 --extra-index-url https://download.pytorch.org/whl/nightly/cpu
  python -c "from datasets import load_dataset;import soundfile as sf;sample = load_dataset('distil-whisper/librispeech_long', 'clean', split='validation')[0]['audio'];sf.write('${MODEL_DIR}/$AUDIO_FILE', sample['array'][:sample['sampling_rate']*30], sample['sampling_rate'])"
fi

ls -al
echo "::endgroup::"

echo "::group::Build $MODEL_NAME Runner"

if [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "metal" ]; then
  echo "Error: Unsupported device '$DEVICE'. Must be 'cuda' or 'metal'."
  exit 1
fi

MAKE_TARGET="${RUNNER_PATH}-${DEVICE}"
make "${MAKE_TARGET}"
echo "::endgroup::"

echo "::group::Run $MODEL_NAME Runner"
set +e
if [ "$DEVICE" = "cuda" ]; then
  export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
fi

# Build runner command with common arguments
RUNNER_BIN="cmake-out/examples/models/$RUNNER_PATH/$RUNNER_TARGET"
RUNNER_ARGS="--model_path ${MODEL_DIR}/model.pte --temperature 0"
# For CUDA, add data_path argument (Metal embeds data in .pte)
if [ "$DEVICE" = "cuda" ]; then
  RUNNER_ARGS="$RUNNER_ARGS --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd"
fi

# Add model-specific arguments
case "$MODEL_NAME" in
  voxtral)
    RUNNER_ARGS="$RUNNER_ARGS --tokenizer_path ${MODEL_DIR}/$TOKENIZER_FILE --audio_path ${MODEL_DIR}/$AUDIO_FILE --processor_path ${MODEL_DIR}/$PREPROCESSOR"
    ;;
  whisper-*)
    RUNNER_ARGS="$RUNNER_ARGS --tokenizer_path ${MODEL_DIR}/ --audio_path ${MODEL_DIR}/$AUDIO_FILE --processor_path ${MODEL_DIR}/$PREPROCESSOR"
    ;;
  gemma3)
    RUNNER_ARGS="$RUNNER_ARGS --tokenizer_path ${MODEL_DIR}/ --image_path $IMAGE_PATH"
    ;;
  parakeet)
    RUNNER_ARGS="--model_path ${MODEL_DIR}/model.pte --audio_path ${MODEL_DIR}/$AUDIO_FILE --tokenizer_path ${MODEL_DIR}/$TOKENIZER_FILE"
    # For CUDA, add data_path argument (Metal embeds data in .pte)
    if [ "$DEVICE" = "cuda" ]; then
      RUNNER_ARGS="$RUNNER_ARGS --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd"
    fi
    ;;
esac

OUTPUT=$($RUNNER_BIN $RUNNER_ARGS 2>&1)
EXIT_CODE=$?
set -e

echo "Runner output:"
echo "$OUTPUT"

if [ $EXIT_CODE -ne 0 ]; then
  echo "Unexpected exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi

# Validate output for models that have expected output
if [ -n "$EXPECTED_OUTPUT" ]; then
  if ! echo "$OUTPUT" | grep -iq "$EXPECTED_OUTPUT"; then
    echo "Expected output '$EXPECTED_OUTPUT' not found in output"
    exit 1
  else
    echo "Success: '$EXPECTED_OUTPUT' found in output"
  fi
else
  echo "SUCCESS: Runner completed successfully"
fi
echo "::endgroup::"

popd
