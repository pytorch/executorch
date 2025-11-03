#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Export model to CUDA format with optional quantization

show_help() {
  cat << EOF
Usage: export_model_cuda_artifact.sh <hf_model> [quant_name] [output_dir]

Export a HuggingFace model to CUDA format with optional quantization.

Arguments:
  hf_model     HuggingFace model ID (required)
               Supported models:
                 - mistralai/Voxtral-Mini-3B-2507
                 - openai/whisper-small
                 - google/gemma-3-4b-it

  quant_name   Quantization type (optional, default: non-quantized)
               Options:
                 - non-quantized
                 - quantized-int4-tile-packed
                 - quantized-int4-weight-only

  output_dir   Output directory for artifacts (optional, default: current directory)

Examples:
  export_model_cuda_artifact.sh "openai/whisper-small"
  export_model_cuda_artifact.sh "mistralai/Voxtral-Mini-3B-2507" "quantized-int4-tile-packed"
  export_model_cuda_artifact.sh "google/gemma-3-4b-it" "non-quantized" "./output"
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

set -eux

HF_MODEL="$1"
QUANT_NAME="${2:-non-quantized}"
OUTPUT_DIR="${3:-.}"

# Determine model configuration based on HF model ID
case "$HF_MODEL" in
  mistralai/Voxtral-Mini-3B-2507)
    MODEL_NAME="voxtral"
    TASK="multimodal-text-to-text"
    MAX_SEQ_LEN="1024"
    EXTRA_PIP="mistral-common librosa"
    PREPROCESSOR_FEATURE_SIZE="128"
    PREPROCESSOR_OUTPUT="voxtral_preprocessor.pte"
    ;;
  openai/whisper-small)
    MODEL_NAME="whisper"
    TASK="automatic-speech-recognition"
    MAX_SEQ_LEN=""
    EXTRA_PIP="librosa"
    PREPROCESSOR_FEATURE_SIZE="80"
    PREPROCESSOR_OUTPUT="whisper_preprocessor.pte"
    ;;
  google/gemma-3-4b-it)
    MODEL_NAME="gemma3"
    TASK="multimodal-text-to-text"
    MAX_SEQ_LEN="64"
    EXTRA_PIP=""
    PREPROCESSOR_FEATURE_SIZE=""
    PREPROCESSOR_OUTPUT=""
    ;;
  *)
    echo "Error: Unsupported model '$HF_MODEL'"
    echo "Supported models: mistralai/Voxtral-Mini-3B-2507, openai/whisper-small, google/gemma-3-4b-it"
    exit 1
    ;;
esac

# Determine quantization args based on quant name
case "$QUANT_NAME" in
  non-quantized)
    EXTRA_ARGS=""
    ;;
  quantized-int4-tile-packed)
    EXTRA_ARGS="--qlinear 4w --qlinear_encoder 4w --qlinear_packing_format tile_packed_to_4d --qlinear_encoder_packing_format tile_packed_to_4d"
    ;;
  quantized-int4-weight-only)
    EXTRA_ARGS="--qlinear_encoder 4w"
    ;;
  *)
    echo "Error: Unsupported quantization '$QUANT_NAME'"
    echo "Supported quantizations: non-quantized, quantized-int4-tile-packed, quantized-int4-weight-only"
    exit 1
    ;;
esac

echo "::group::Export $MODEL_NAME"

if [ -n "$EXTRA_PIP" ]; then
  pip install $EXTRA_PIP
fi
pip list

MAX_SEQ_LEN_ARG=""
if [ -n "$MAX_SEQ_LEN" ]; then
  MAX_SEQ_LEN_ARG="--max_seq_len $MAX_SEQ_LEN"
fi
optimum-cli export executorch \
    --model "$HF_MODEL" \
    --task "$TASK" \
    --recipe "cuda" \
    --dtype bfloat16 \
    --device cuda \
    ${MAX_SEQ_LEN_ARG} \
    ${EXTRA_ARGS} \
    --output_dir ./

if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  python -m executorch.extension.audio.mel_spectrogram \
      --feature_size $PREPROCESSOR_FEATURE_SIZE \
      --stack_output \
      --max_audio_len 300 \
      --output_file $PREPROCESSOR_OUTPUT
fi

test -f model.pte
test -f aoti_cuda_blob.ptd
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  test -f $PREPROCESSOR_OUTPUT
fi
echo "::endgroup::"

echo "::group::Store $MODEL_NAME Artifacts"
mkdir -p "${OUTPUT_DIR}"
cp model.pte "${OUTPUT_DIR}/"
cp aoti_cuda_blob.ptd "${OUTPUT_DIR}/"
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  cp $PREPROCESSOR_OUTPUT "${OUTPUT_DIR}/"
fi
ls -al "${OUTPUT_DIR}"
echo "::endgroup::"
