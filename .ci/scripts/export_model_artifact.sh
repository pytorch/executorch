#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Export model to CUDA/Metal/XNNPACK format with optional quantization

show_help() {
  cat << EOF
Usage: export_model_artifact.sh <device> <hf_model> [quant_name] [output_dir] [mode]

Export a HuggingFace model to CUDA/Metal/XNNPACK format with optional quantization.

Arguments:
  device       cuda, metal, or xnnpack (required)

  hf_model     HuggingFace model ID (required)
               Supported models:
                 - mistralai/Voxtral-Mini-3B-2507
                 - mistralai/Voxtral-Mini-4B-Realtime-2602
                 - openai/whisper series (whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo})
                 - google/gemma-3-4b-it
                 - nvidia/parakeet-tdt

  quant_name   Quantization type (optional, default: non-quantized)
               Options:
                 - non-quantized
                 - quantized-int4-tile-packed (CUDA only)
                 - quantized-int4-weight-only (CUDA only)
                 - quantized-int4-metal (Metal only)
                 - quantized-8da4w (XNNPACK only)

  output_dir   Output directory for artifacts (optional, default: current directory)

  mode         Export mode (optional, default: auto-detect based on model and device)
               Supported modes:
                 - vr-streaming: Voxtral Realtime streaming mode
                 - vr-offline: Voxtral Realtime offline mode

Examples:
  export_model_artifact.sh metal "openai/whisper-small"
  export_model_artifact.sh metal "nvidia/parakeet-tdt" "quantized-int4-metal"
  export_model_artifact.sh metal "mistralai/Voxtral-Mini-4B-Realtime-2602" "quantized-int4-metal"
  export_model_artifact.sh metal "mistralai/Voxtral-Mini-4B-Realtime-2602" "non-quantized" "." "vr-streaming"
  export_model_artifact.sh cuda "mistralai/Voxtral-Mini-3B-2507" "quantized-int4-tile-packed"
  export_model_artifact.sh cuda "google/gemma-3-4b-it" "non-quantized" "./output"
  export_model_artifact.sh cuda "nvidia/parakeet-tdt" "non-quantized" "./output"
  export_model_artifact.sh xnnpack "nvidia/parakeet-tdt" "quantized-8da4w" "./output"
  export_model_artifact.sh xnnpack "mistralai/Voxtral-Mini-4B-Realtime-2602" "quantized-8da4w" "./output"
  export_model_artifact.sh xnnpack "mistralai/Voxtral-Mini-4B-Realtime-2602" "non-quantized" "./output" "vr-offline"
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
MODE="${5:-}"

# Validate mode if specified
if [ -n "$MODE" ]; then
  case "$MODE" in
    vr-streaming|vr-offline)
      # Voxtral Realtime modes require Voxtral Realtime model
      if [ "$HF_MODEL" != "mistralai/Voxtral-Mini-4B-Realtime-2602" ]; then
        echo "Error: Mode '$MODE' can only be used with Voxtral Realtime model"
        echo "Provided model: $HF_MODEL"
        exit 1
      fi
      ;;
    *)
      echo "Error: Unsupported mode '$MODE'"
      echo "Supported modes: vr-streaming, vr-offline"
      exit 1
      ;;
  esac
fi

case "$DEVICE" in
  cuda)
    ;;
  cuda-windows)
    ;;
  metal)
    ;;
  xnnpack)
    ;;
  *)
    echo "Error: Unsupported device '$DEVICE'"
    echo "Supported devices: cuda, cuda-windows, metal, xnnpack"
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
    MODEL_NAME="parakeet"
    TASK=""
    MAX_SEQ_LEN=""
    EXTRA_PIP=""
    PREPROCESSOR_FEATURE_SIZE=""
    PREPROCESSOR_OUTPUT=""
    ;;
  mistralai/Voxtral-Mini-4B-Realtime-2602)
    MODEL_NAME="voxtral_realtime"
    TASK=""
    MAX_SEQ_LEN=""
    EXTRA_PIP="mistral-common librosa"
    PREPROCESSOR_FEATURE_SIZE=""
    PREPROCESSOR_OUTPUT=""
    ;;
  *)
    echo "Error: Unsupported model '$HF_MODEL'"
    echo "Supported models: mistralai/Voxtral-Mini-3B-2507, mistralai/Voxtral-Mini-4B-Realtime-2602, openai/whisper-{small, medium, large, large-v2, large-v3, large-v3-turbo}, google/gemma-3-4b-it, nvidia/parakeet-tdt"
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
      echo "Error: Metal backend does not support quantization '$QUANT_NAME'"
      exit 1
    fi
    EXTRA_ARGS="--qlinear 4w --qlinear_encoder 4w --qlinear_packing_format tile_packed_to_4d --qlinear_encoder_packing_format tile_packed_to_4d"
    ;;
  quantized-int4-weight-only)
    if [ "$DEVICE" = "metal" ]; then
      echo "Error: Metal backend does not support quantization '$QUANT_NAME'"
      exit 1
    fi
    EXTRA_ARGS="--qlinear_encoder 4w"
    ;;
  quantized-int4-metal)
    if [ "$DEVICE" != "metal" ]; then
      echo "Error: Quantization '$QUANT_NAME' only supported on Metal backend"
      exit 1
    fi
    EXTRA_ARGS="--qlinear fpa4w --qlinear_encoder fpa4w"
    ;;
  quantized-8da4w)
    if [ "$DEVICE" != "xnnpack" ]; then
      echo "Error: quantized-8da4w is only supported with xnnpack device"
      exit 1
    fi
    EXTRA_ARGS="--qlinear 8da4w --qlinear_group_size 32 --qlinear_encoder 8da4w --qlinear_encoder_group_size 32"
    ;;
  *)
    echo "Error: Unsupported quantization '$QUANT_NAME'"
    echo "Supported quantizations: non-quantized, quantized-int4-tile-packed, quantized-int4-weight-only, quantized-int4-metal, quantized-8da4w"
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

  # Set dtype based on backend (XNNPACK uses fp32, CUDA/Metal use bf16)
  if [ "$DEVICE" = "xnnpack" ]; then
    DTYPE_ARG=""
  else
    DTYPE_ARG="--dtype bf16"
  fi

  python -m executorch.examples.models.parakeet.export_parakeet_tdt \
      --backend "$DEVICE" \
      --output-dir "${OUTPUT_DIR}" \
      ${DTYPE_ARG} \
      ${EXTRA_ARGS}

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

# Voxtral Realtime uses a custom export script
if [ "$MODEL_NAME" = "voxtral_realtime" ]; then
  pip install safetensors huggingface_hub

  # Download model weights from HuggingFace (requires HF_TOKEN for gated model)
  LOCAL_MODEL_DIR="${OUTPUT_DIR}/model_weights"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_MODEL}', local_dir='${LOCAL_MODEL_DIR}')"

  # Per-component quantization flags
  VR_QUANT_ARGS=""
  if [ "$QUANT_NAME" = "quantized-8da4w" ]; then
    VR_QUANT_ARGS="--qlinear-encoder 8da4w --qlinear 8da4w --qlinear-group-size 32 --qembedding 8w"
  elif [ "$QUANT_NAME" = "quantized-int4-metal" ]; then
    VR_QUANT_ARGS="--qlinear-encoder fpa4w --qlinear fpa4w"
  fi

  # Determine streaming mode based on MODE parameter
  USE_STREAMING="false"
  if [ "$MODE" = "vr-streaming" ]; then
    USE_STREAMING="true"
  elif [ "$MODE" = "vr-offline" ]; then
    USE_STREAMING="false"
  elif [ -z "$MODE" ]; then
    # Auto-detect: XNNPACK uses streaming, others use offline
    if [ "$DEVICE" = "xnnpack" ]; then
      USE_STREAMING="true"
    fi
  fi

  # Configure export and preprocessor based on streaming mode
  STREAMING_ARG=""
  PREPROCESSOR_ARGS="--feature_size 128 --output_file ${OUTPUT_DIR}/preprocessor.pte"
  if [ "$USE_STREAMING" = "true" ]; then
    STREAMING_ARG="--streaming"
    PREPROCESSOR_ARGS="$PREPROCESSOR_ARGS --streaming"
  else
    PREPROCESSOR_ARGS="$PREPROCESSOR_ARGS --stack_output --max_audio_len 300"
  fi

  python -m executorch.examples.models.voxtral_realtime.export_voxtral_rt \
      --model-path "$LOCAL_MODEL_DIR" \
      --backend "$DEVICE" \
      ${STREAMING_ARG} \
      --output-dir "${OUTPUT_DIR}" \
      ${VR_QUANT_ARGS}

  # Export preprocessor
  python -m executorch.extension.audio.mel_spectrogram ${PREPROCESSOR_ARGS}

  test -f "${OUTPUT_DIR}/model.pte"
  test -f "${OUTPUT_DIR}/preprocessor.pte"
  # Copy tokenizer from downloaded model weights
  cp "$LOCAL_MODEL_DIR/tekken.json" "${OUTPUT_DIR}/tekken.json"
  ls -al "${OUTPUT_DIR}"
  echo "::endgroup::"
  exit 0
fi

MAX_SEQ_LEN_ARG=""
if [ -n "$MAX_SEQ_LEN" ]; then
  MAX_SEQ_LEN_ARG="--max_seq_len $MAX_SEQ_LEN"
fi

DEVICE_ARG=""
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "cuda-windows" ]; then
  DEVICE_ARG="--device cuda"
elif [ "$DEVICE" = "metal" ]; then
  DEVICE_ARG="--device mps"
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

# Determine blob file name - cuda and cuda-windows both use aoti_cuda_blob.ptd
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "cuda-windows" ]; then
  BLOB_FILE="aoti_cuda_blob.ptd"
else
  BLOB_FILE="aoti_${DEVICE}_blob.ptd"
fi

test -f model.pte
# CUDA saves named data to separate .ptd file, Metal embeds in .pte
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "cuda-windows" ]; then
  test -f $BLOB_FILE
fi
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  test -f $PREPROCESSOR_OUTPUT
fi
echo "::endgroup::"

echo "::group::Store $MODEL_NAME Artifacts"
mkdir -p "${OUTPUT_DIR}"
mv model.pte "${OUTPUT_DIR}/"
# CUDA saves named data to separate .ptd file, Metal embeds in .pte
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "cuda-windows" ]; then
  mv $BLOB_FILE "${OUTPUT_DIR}/"
fi
if [ -n "$PREPROCESSOR_OUTPUT" ]; then
  mv $PREPROCESSOR_OUTPUT "${OUTPUT_DIR}/"
fi
ls -al "${OUTPUT_DIR}"
echo "::endgroup::"
