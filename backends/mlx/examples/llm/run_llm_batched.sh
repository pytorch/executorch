#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build and run the batched MLX LLM example.
#
#   PTE=/path/model.pte TOKENIZER=/path/tokenizer.json \
#     ./run_llm_batched.sh "prompt one" "prompt two"
#   PTE=... TOKENIZER=... ./run_llm_batched.sh --skip-build
#
# The .pte must have been exported with --use-offgraph-cache: the executor
# reads the KV layout the export publishes and builds a cell cache from it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXAMPLE_SRC="$REPO/backends/mlx/examples/llm"
EXAMPLE_BUILD="$REPO/cmake-out/backends/mlx/examples/llm"
BINARY="$EXAMPLE_BUILD/mlx_run_llm_batched"
JOBS="$(( $(sysctl -n hw.ncpu) - 1 ))"

PTE="${PTE:-}"
TOKENIZER="${TOKENIZER:-}"
MAX_SESSION_TOKENS="${MAX_SESSION_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
FLUSH_EVERY="${FLUSH_EVERY:-8}"
# Sessions to submit. Empty uses each prompt once; a number cycles through the
# prompts to reach it, so concurrency can be swept without editing the list.
NUM_SEQUENCES="${NUM_SEQUENCES:-}"
OUT_PREFIX="${OUT_PREFIX:-/tmp/batched_gen}"

skip_build=0
prompts=()
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) skip_build=1 ;;
    --num-sequences)
      if [ $# -lt 2 ]; then
        echo "--num-sequences needs a value" >&2
        exit 1
      fi
      NUM_SEQUENCES="$2"
      shift
      ;;
    --num-sequences=*) NUM_SEQUENCES="${1#--num-sequences=}" ;;
    *) prompts+=("$1") ;;
  esac
  shift
done
if [ ${#prompts[@]} -eq 0 ]; then
  # A passage repeated into a long context. The scheduler splits it across
  # multiple prefill chunks, exercising a path single-sentence prompts do not.
  passage="Aldermere stood where the Vell and the Corrin met, and the two \
rivers never mixed: the Vell ran grey with silt off the northern moors while \
the Corrin came down clear and cold from the Iron Steps, and for a mile below \
the confluence a traveller could see the seam between them. The bridge guilds \
grew rich on that seam. The Coopers held the eastern span, the Ferriers the \
western, and between them the Assay House weighed every cargo twice, once on \
each bank, and charged for the discrepancy."
  long_context=""
  for _ in $(seq 18); do long_context+="$passage "; done

  prompts=(
    "Write a long history of astronomy."
    "Explain quantum computing in great detail."
    "Write an extensive guide to gardening."
    "Tell a long story about an ocean voyage."
    "Describe the evolution of operating systems in depth."
    "Write a comprehensive introduction to economics."
    "Explain how cities develop over centuries in detail."
    "Write a long biography of a fictional inventor."
    "Write a detailed, numbered list of 20 practical tips for staying productive while working from home. Explain each tip in two or three sentences."
    "Explain how a computer works, starting from transistors and logic gates and building all the way up to running a modern application. Cover every layer in detail."
    "Write a complete beginner's tutorial for building a REST API in Python with Flask. Include setup, code examples for each endpoint, error handling, testing, and deployment."
    "Write a long adventure story, in at least five chapters, about a cartographer who finds a map of a place that does not exist. Include dialogue and chapter headings."
    "Write a 16-week training plan for a first marathon. Give every week its own section covering mileage, key workouts, nutrition, and recovery."
    "Trace the history of written language from cuneiform to the printing press. Give each major writing system its own section covering origin, mechanics, and cultural impact."
    "Write the complete manual for a command-line tool called flux that manages database migrations. Document every subcommand and flag, with examples and a troubleshooting section."
    "Compare the twelve most influential programming languages. Give each its own section covering origins, design goals, strengths, weaknesses, and lasting legacy."
    # Long prefill, short generation: cost is almost all prompt processing.
    "${long_context}

Summarize the passage above in exactly three sentences."
    # Long prefill and long generation, so one session holds a wide context
    # while it decodes.
    "${long_context}

Using only the passage above, write a detailed guide for a merchant arriving in Aldermere. Cover the rivers, the bridge guilds, and the Assay House, with a section on each."
  )
fi

if [ -z "$PTE" ] || [ -z "$TOKENIZER" ]; then
  echo "set PTE=/path/model.pte and TOKENIZER=/path/tokenizer-file" >&2
  exit 1
fi
for f in "$PTE" "$TOKENIZER"; do
  if [ ! -f "$f" ]; then
    echo "missing: $f" >&2
    exit 1
  fi
done

if [ "$skip_build" -eq 0 ]; then
  echo "==> installing ExecuTorch (mlx-release)"
  cd "$REPO"
  cmake --preset mlx-release
  cmake --build cmake-out --target install -j"$JOBS"

  echo "==> building mlx_run_llm_batched"
  cmake -S "$EXAMPLE_SRC" -B "$EXAMPLE_BUILD" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$EXAMPLE_BUILD" --target mlx_run_llm_batched -j"$JOBS"
fi

if [ ! -x "$BINARY" ]; then
  echo "no binary at $BINARY; run without --skip-build" >&2
  exit 1
fi

# Resize the prompt list to the requested session count. Cycling rather than
# only truncating keeps the mix of lengths intact as the count grows, which
# matters because a batch's cost depends on the spread of its sessions and not
# just how many there are.
if [ -n "$NUM_SEQUENCES" ]; then
  case "$NUM_SEQUENCES" in
    ''|*[!0-9]*)
      echo "--num-sequences takes a non-negative integer" >&2
      exit 1
      ;;
  esac
  if [ "$NUM_SEQUENCES" -eq 0 ]; then
    echo "--num-sequences must be at least 1" >&2
    exit 1
  fi
  selected=()
  i=0
  while [ "$i" -lt "$NUM_SEQUENCES" ]; do
    selected+=("${prompts[$((i % ${#prompts[@]}))]}")
    i=$((i + 1))
  done
  prompts=("${selected[@]}")
fi

echo "==> running ${#prompts[@]} prompts (batched-cell cache)"
"$BINARY" \
  --pte "$PTE" --metrics --max_decode_sequences 16 \
  --tokenizer "$TOKENIZER" \
  --max_session_tokens "$MAX_SESSION_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --flush_every "$FLUSH_EVERY" \
  --out_prefix "$OUT_PREFIX" \
  "${prompts[@]}"

# echo
# for i in "${!prompts[@]}"; do
#   out="${OUT_PREFIX}_${i}.txt"
#   echo "--- [$i] ${prompts[$i]}"
#   [ -f "$out" ] && cat "$out" || echo "(no output)"
#   echo
# done
