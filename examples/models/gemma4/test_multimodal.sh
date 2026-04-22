#!/usr/bin/env bash
# Gemma4 multimodal runner — end-to-end test script
#
# Single .pte (gemma4_multimodal_v9.pte) serves all 3 modalities via
# ExecuTorch's standard MultimodalRunner (create_multimodal_runner).
#
# Tests:
#   1. Text-only   — prompt only, no image/audio
#   2. Image+text  — image.jpg + prompt
#   3. Audio+text  — obama_short20.wav (20s speech) + prompt
#   4. Short audio — 2s clip (tests model behavior on minimal audio context)
#
# Usage:
#   ./test_multimodal.sh                      # run all tests
#   ./test_multimodal.sh --text-only          # only text tests
#   ./test_multimodal.sh --multimodal         # only image+audio tests
#   ./test_multimodal.sh --quick              # 1 test per modality, short seq_len

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit if your pte/model locations differ
# ---------------------------------------------------------------------------
EXECUTORCH_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNNER="$EXECUTORCH_ROOT/cmake-out/examples/models/gemma4/gemma4_runner"
MODEL="/tmp/gemma4_multimodal_v9.pte"   # single .pte for all 3 modalities
TOKENIZER="$HOME/models/gemma-4-E2B-it/tokenizer.json"
IMAGE="$EXECUTORCH_ROOT/image.jpg"
AUDIO_20S="$EXECUTORCH_ROOT/obama_short20.wav"
AUDIO_2S="/tmp/gemma4_test_2s.wav"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
RUN_TEXT=true
RUN_MM=true
QUICK=false

for arg in "$@"; do
  case "$arg" in
    --text-only)  RUN_MM=false ;;
    --multimodal) RUN_TEXT=false ;;
    --quick)      QUICK=true ;;
    --help|-h)
      sed -n '3,8p' "$0" | sed 's/^# //'
      exit 0 ;;
  esac
done

SEQ_TEXT=20;   SEQ_IMG=50;  SEQ_AUDIO=80
$QUICK && { SEQ_TEXT=8; SEQ_IMG=15; SEQ_AUDIO=20; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0; FAIL=0; SKIP=0

run_test() {
  local name="$1"; shift
  local timeout_s="$1"; shift

  printf "\n\033[1;34m─── %s ───\033[0m\n" "$name"
  local out
  if out=$(timeout "$timeout_s" "$@" 2>/dev/null); then
    # Strip PyTorchObserver JSON line for cleaner display
    local text; text=$(echo "$out" | grep -v '^PyTorchObserver')
    local stats; stats=$(echo "$out" | grep '^PyTorchObserver' | \
      python3 -c "
import sys, json
line = sys.stdin.read().strip()
if not line: sys.exit(0)
d = json.loads(line)
print(f\"  prefill={d.get('prefill_token_per_sec',0):.0f} tok/s  decode={d.get('decode_token_per_sec',0):.1f} tok/s  tokens={d.get('prompt_tokens',0)}+{d.get('generated_tokens',0)}\")
" 2>/dev/null || true)
    printf "\033[0;32mOutput:\033[0m %s\n" "$text"
    [ -n "$stats" ] && printf "\033[0;90m%s\033[0m\n" "$stats"
    printf "\033[0;32m✓ PASS\033[0m\n"
    PASS=$((PASS + 1))
  else
    local rc=$?
    if [ $rc -eq 124 ]; then
      printf "\033[0;33m⚠ TIMEOUT (%ss)\033[0m\n" "$timeout_s"
    else
      printf "\033[0;31m✗ FAIL (exit %d)\033[0m\n" "$rc"
    fi
    FAIL=$((FAIL + 1))
  fi
}

skip_test() {
  local name="$1"; local reason="$2"
  printf "\n\033[1;34m─── %s ───\033[0m\n" "$name"
  printf "\033[0;33m⊘ SKIP: %s\033[0m\n" "$reason"
  SKIP=$((SKIP + 1))
}

check_file() {
  [ -f "$1" ] || { echo "MISSING: $1" >&2; return 1; }
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
printf "\n\033[1mGemma4 multimodal test suite\033[0m\n"
printf "Runner : %s\n" "$RUNNER"
printf "Model  : %s\n" "$MODEL"
printf "Tokens : %s\n" "$TOKENIZER"

check_file "$RUNNER"  || { echo "Build the runner first: cmake --build cmake-out/examples/models/gemma4"; exit 1; }
check_file "$TOKENIZER" || exit 1

# Create 2s clip from the 20s audio (first 2s = 32000 samples at 16kHz)
if [ -f "$AUDIO_20S" ] && ! [ -f "$AUDIO_2S" ]; then
  python3 -c "
import wave
src = wave.open('$AUDIO_20S', 'rb')
dst = wave.open('$AUDIO_2S', 'wb')
dst.setnchannels(src.getnchannels())
dst.setsampwidth(src.getsampwidth())
dst.setframerate(src.getframerate())
dst.writeframes(src.readframes(2 * src.getframerate()))
src.close(); dst.close()
print('Created $AUDIO_2S')
" || true
fi

# ---------------------------------------------------------------------------
# TEXT-ONLY tests (all use the multimodal pte — single artifact for all modes)
# ---------------------------------------------------------------------------
if $RUN_TEXT; then
  printf "\n\033[1m== TEXT-ONLY ==\033[0m\n"

  if check_file "$MODEL" 2>/dev/null; then
    run_test "Text: capital of France" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --prompt "What is the capital of France?" --seq_len $SEQ_TEXT

    run_test "Text: math" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --prompt "What is 12 multiplied by 8?" --seq_len $SEQ_TEXT

    $QUICK || run_test "Text: code generation" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --prompt "Write a Python function to reverse a string." --seq_len 40

    $QUICK || run_test "Text: general knowledge" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --prompt "Explain what a neural network is in one sentence." --seq_len 30
  else
    skip_test "Text tests" "MM pte not found at $MODEL"
  fi
fi

# ---------------------------------------------------------------------------
# MULTIMODAL tests
# ---------------------------------------------------------------------------
if $RUN_MM; then
  printf "\n\033[1m== IMAGE + TEXT ==\033[0m\n"

  if check_file "$MODEL" 2>/dev/null && check_file "$IMAGE" 2>/dev/null; then
    run_test "Image: describe" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --image_path "$IMAGE" \
      --prompt "Describe this image." --seq_len $SEQ_IMG

    $QUICK || run_test "Image: what color" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --image_path "$IMAGE" \
      --prompt "What are the dominant colors in this image?" --seq_len $SEQ_IMG

    $QUICK || run_test "Image: type of scene" 120 \
      "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
      --image_path "$IMAGE" \
      --prompt "Is this a portrait, landscape, or something else?" --seq_len $SEQ_IMG
  else
    skip_test "Image+text tests" "MM pte or image.jpg not found"
  fi

  printf "\n\033[1m== AUDIO + TEXT ==\033[0m\n"

  if check_file "$MODEL" 2>/dev/null; then
    if check_file "$AUDIO_2S" 2>/dev/null; then
      run_test "Audio: 2s clip — what sound" 60 \
        "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
        --audio_path "$AUDIO_2S" \
        --prompt "What do you hear?" --seq_len $SEQ_AUDIO
    else
      skip_test "Audio 2s clip" "$AUDIO_2S not found"
    fi

    if check_file "$AUDIO_20S" 2>/dev/null; then
      run_test "Audio: 20s speech — what is being said" 180 \
        "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
        --audio_path "$AUDIO_20S" \
        --prompt "What is being said?" --seq_len $SEQ_AUDIO

      $QUICK || run_test "Audio: 20s speech — transcribe" 180 \
        "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
        --audio_path "$AUDIO_20S" \
        --prompt "Transcribe this audio." --seq_len 120

      $QUICK || run_test "Audio: 20s speech — music or speech" 180 \
        "$RUNNER" --model_path "$MODEL" --tokenizer_path "$TOKENIZER" \
        --audio_path "$AUDIO_20S" \
        --prompt "Is this music or speech? Explain." --seq_len $SEQ_AUDIO
    else
      skip_test "Audio 20s tests" "$AUDIO_20S not found"
    fi
  else
    skip_test "Audio+text tests" "MM pte not found at $MODEL"
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL + SKIP))
printf "\n\033[1m══════════════════════════════\033[0m\n"
printf "\033[1mResults: %d/%d passed\033[0m" "$PASS" "$TOTAL"
[ $SKIP -gt 0 ] && printf "  (%d skipped)" "$SKIP"
printf "\n"
[ $FAIL -gt 0 ] && printf "\033[0;31m%d test(s) failed\033[0m\n" "$FAIL"
printf "\033[1m══════════════════════════════\033[0m\n\n"

[ $FAIL -eq 0 ]
