#!/usr/bin/env bash
# Drive the 3-config coopmat benchmark on the phone over adb.
#
#   B-coopmat : buffer PTE, coopmat on   (your contribution)
#   B-tiled   : buffer PTE, ET_VK_DISABLE_COOPMAT=1 (fair, same-storage baseline)
#   T-tiled   : texture PTE (default ExecuTorch baseline)
#
# Report the 3-way: kernel gain = B-coopmat vs B-tiled; storage penalty =
# B-tiled vs T-tiled; e2e = B-coopmat vs T-tiled. coopmat only affects PREFILL
# (decode is gemv, M=1). Use a long prompt to make prefill dominate.
#
# Usage:
#   bench_phone.sh <buffer_pte> <texture_pte> [prompt] [seq_len]
# Example:
#   bench_phone.sh llama3_1_8b_4w_buffer.pte llama3_1_8b_4w_texture.pte \
#       "The history of computing began" 96
#
# Prereqs on host: adb device visible (see memory: adb-device-sj1-box).
# Phone dir layout assumed: $D below. tokenizer.model already pushed.
set -euo pipefail

D=/data/local/tmp/llama_vk
PTE_OUT=/local/yanwen.xu/workspace/pte_out
BIN=llama_main_coopmat          # the rebuilt binary with Plan A C++ changes
TOK=$D/tokenizer.model

BUF_PTE="${1:?buffer pte filename}"
TEX_PTE="${2:?texture pte filename}"
PROMPT="${3:-The history of computing began}"
SEQLEN="${4:-96}"

push() { adb push "$PTE_OUT/$1" "$D/$1" >/dev/null && echo "pushed $1"; }

run() {  # name  env  pte
  local name="$1" env="$2" pte="$3"
  echo "=== $name ($env) ==="
  adb shell "cd $D && $env ./$BIN --model_path=$D/$pte --tokenizer_path=$TOK \
      --prompt='$PROMPT' --seq_len=$SEQLEN --temperature=0 --warmup=true" \
    2>&1 | grep -E "PyTorchObserver|prefill_token|decode_token|tok/s|Error" || true
  echo
}

push "$BUF_PTE"; push "$TEX_PTE"

run "B-coopmat" ""                       "$BUF_PTE"
run "B-tiled"   "ET_VK_DISABLE_COOPMAT=1" "$BUF_PTE"
run "T-tiled"   ""                       "$TEX_PTE"

echo "Done. Parse prefill_token_per_sec from each PyTorchObserver line."
