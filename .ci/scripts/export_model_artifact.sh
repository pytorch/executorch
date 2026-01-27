#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Export model to CUDA/Metal format with optional quantization

show_help() {
  cat << EOF
Usage: export_model_artifact.sh <device> <hf_model> [quant_name] [output_dir]

Export a HuggingFace model to CUDA/Metal format with optional quantization.

Arguments:
  device       cuda or metal (required)

  hf_model     HuggingFace model ID (required)
               Supported models:
                 - mistralai/Voxtral-Mini-3B-2507
                 - openai/whisper series (whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo})
                 - google/gemma-3-4b-it
                 - nvidia/parakeet-tdt

  quant_name   Quantization type (optional, default: non-quantized)
               Options:
                 - non-quantized
                 - quantized-int4-tile-packed
                 - quantized-int4-weight-only

  output_dir   Output directory for artifacts (optional, default: current directory)

Examples:
  export_model_artifact.sh metal "openai/whisper-small"
  export_model_artifact.sh cuda "mistralai/Voxtral-Mini-3B-2507" "quantized-int4-tile-packed"
  export_model_artifact.sh cuda "google/gemma-3-4b-it" "non-quantized" "./output"
  export_model_artifact.sh cuda "nvidia/parakeet-tdt" "non-quantized" "./output"
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

DEVICE="$1"
HF_MODEL="$2"
QUANT_NAME="${3:-non-quantized}"
OUTPUT_DIR="${4:-.}"

case "$DEVICE" in
  cuda)
    ;;
  metal)
    ;;
  *)
    echo "Error: Unsupported device '$DEVICE'"
    echo "Supported devices: cuda, metal"
    exit 1
    ;;
esac

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
  openai/whisper-*)
    MODEL_NAME="whisper"
    TASK="automatic-speech-recognition"
    MAX_SEQ_LEN=""
    EXTRA_PIP="librosa"
    PREPROCESSOR_OUTPUT="whisper_preprocessor.pte"
    if [[ "$HF_MODEL" == *"large-v3"* ]]; then
      PREPROCESSOR_FEATURE_SIZE="128"
    else
      PREPROCESSOR_FEATURE_SIZE="80"
    fi
    ;;
  google/gemma-3-4b-it)
    if [ "$DEVICE" = "metal" ]; then
      echo "Error: Export for device 'metal' is not yet tested for model '$HF_MODEL'"
      exit 1
    fi
    MODEL_NAME="gemma3"
    TASK="multimodal-text-to-text"
    MAX_SEQ_LEN="64"
    EXTRA_PIP=""
    PREPROCESSOR_FEATURE_SIZE=""
    PREPROCESSOR_OUTPUT=""
    ;;
  nvidia/parakeet-tdt)
    if [ "$DEVICE" = "metal" ]; then
      echo "Error: Export for device 'metal' is not yet tested for model '$HF_MODEL'"
      exit 1
    fi
    MODEL_NAME="parakeet"
    TASK=""
    MAX_SEQ_LEN=""
    EXTRA_PIP=""
    PREPROCESSOR_FEATURE_SIZE=""
    PREPROCESSOR_OUTPUT=""
    ;;
  *)
    echo "Error: Unsupported model '$HF_MODEL'"
    echo "Supported models: mistralai/Voxtral-Mini-3B-2507, openai/whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo}, google/gemma-3-4b-it, nvidia/parakeet-tdt"
    exit 1
    ;;
esac

# Determine quantization args based on quant name
case "$QUANT_NAME" in
  non-quantized)
    EXTRA_ARGS=""
    ;;
  quantized-int4-tile-packed)
    if [ "$DEVICE" = "metal" ]; then
      echo "Error: Metal backend does not yet support quantization '$QUANT_NAME'"
      exit 1
    fi
    EXTRA_ARGS="--qlinear 4w --qlinear_encoder 4w --qlinear_packing_format tile_packed_to_4d --qlinear_encoder_packing_format tile_packed_to_4d"
    ;;
  quantized-int4-weight-only)
    if [ "$DEVICE" = "metal" ]; then
      echo "Error: Metal backend does not yet support quantization '$QUANT_NAME'"
      exit 1
    fi
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

# Parakeet uses a custom export script
if [ "$MODEL_NAME" = "parakeet" ]; then
  pip install -r examples/models/parakeet/install_requirements.txt

  python examples/models/parakeet/export_parakeet_tdt.py \
      --backend "$DEVICE" \
      --output-dir "${OUTPUT_DIR}" \
      --dtype bf16

  test -f "${OUTPUT_DIR}/model.pte"
  # CUDA saves named data to separate .ptd file, Metal embeds in .pte
  if [ "$DEVICE" = "cuda" ]; then
    test -f "${OUTPUT_DIR}/aoti_cuda_blob.ptd"
  fi
  test -f "${OUTPUT_DIR}/tokenizer.model"
  ls -al "${OUTPUT_DIR}"
  echo "::endgroup::"
  exit 0
fi

MAX_SEQ_LEN_ARG=""
if [ -n "$MAX_SEQ_LEN" ]; then
  MAX_SEQ_LEN_ARG="--max_seq_len $MAX_SEQ_LEN"
fi

DEVICE_ARG=""
if [ "$DEVICE" = "cuda" ]; then
  DEVICE_ARG="--device cuda"
fi

optimum-cli export executorch \
    --model "$HF_MODEL" \
    --task "$TASK" \
    --recipe "$DEVICE" \
    --dtype bfloat16 \
    ${DEVICE_ARG} \
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
# CUDA saves named data to separate .ptd file, Metal embeds in .pte
if [ "$DEVICE" = "cuda" ]; then
  test -f aoti_cuda_blob.ptd
fi
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  test -f $PREPROCESSOR_OUTPUT
fi
echo "::endgroup::"

echo "::group::Store $MODEL_NAME Artifacts"
mkdir -p "${OUTPUT_DIR}"
mv model.pte "${OUTPUT_DIR}/"
# CUDA saves named data to separate .ptd file, Metal embeds in .pte
if [ "$DEVICE" = "cuda" ]; then
  mv aoti_cuda_blob.ptd "${OUTPUT_DIR}/"
fi
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  mv $PREPROCESSOR_OUTPUT "${OUTPUT_DIR}/"
fi
ls -al "${OUTPUT_DIR}"
echo "::endgroup::"
