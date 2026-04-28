#!/bin/bash
# E2E sweep: baseline vs dequant across prompt lengths and generation lengths
set -euo pipefail

RUNNER="${RUNNER:?Set RUNNER to the path of qwen3_5_moe_runner binary}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER to the path of tokenizer.json}"
if [ -n "$LD_PRELOAD_OVERRIDE" ]; then
    export LD_PRELOAD="$LD_PRELOAD_OVERRIDE"
fi

BASELINE="${BASELINE:?Set BASELINE to the baseline model directory}"
DEQUANT="${DEQUANT:?Set DEQUANT to the dequant model directory}"
REPORT_DIR="${REPORT_DIR:-./report_baseline_vs_dequant}"
mkdir -p "$REPORT_DIR"

# Generate prompts of various lengths
gen_prompt() {
    local target=$1
    python3 -c "
base = 'The transformer architecture has revolutionized machine learning and natural language processing by enabling parallel computation across all positions in a sequence, eliminating the sequential bottleneck of recurrent models. '
text = base * ($target // 10 + 5)
print(text[:$target * 6])
"
}

run_one() {
    local label=$1 dir=$2 prompt=$3 max_tok=$4 outfile=$5
    $RUNNER \
      --model_path "$dir/model.pte" \
      --data_path "$dir/aoti_cuda_blob.ptd" \
      --tokenizer_path "$TOKENIZER" \
      --prompt "$prompt" \
      --max_new_tokens "$max_tok" \
      --temperature 0 2>&1 | tee "$outfile"
}

extract_stats() {
    local file=$1
    local ptok=$(grep -oP 'Prompt Tokens: \K\d+' "$file" | tail -1)
    local prate=$(grep -oP 'Prompt evaluation:.*Rate:\s*\K[\d.]+' "$file" | tail -1)
    local gtok=$(grep -oP 'Generated \K\d+' "$file" | tail -1)
    local drate=$(grep -oP 'Generated \d+ tokens:.*Rate:\s*\K[\d.]+' "$file" | tail -1)
    local ttft=$(grep -oP 'Time to first generated token:\s*\K[\d.]+' "$file" | tail -1)
    echo "$ptok,$prate,$gtok,$drate,$ttft"
}

# ============================================================
# Part 1: Performance sweep
# ============================================================
echo "prompt_tokens,gen_tokens,baseline_prefill,baseline_decode,baseline_ttft,dequant_prefill,dequant_decode,dequant_ttft" > "$REPORT_DIR/sweep.csv"

for PTARGET in 128 256 512 1024 2048; do
    PROMPT=$(gen_prompt $PTARGET)
    for GENTOK in 128 256 512; do
        echo "=== P~${PTARGET} G=${GENTOK} ==="

        # Baseline
        BFILE="$REPORT_DIR/run_baseline_p${PTARGET}_g${GENTOK}.txt"
        run_one baseline "$BASELINE" "$PROMPT" "$GENTOK" "$BFILE" > /dev/null 2>&1
        BSTATS=$(extract_stats "$BFILE")

        # Dequant
        DFILE="$REPORT_DIR/run_dequant_p${PTARGET}_g${GENTOK}.txt"
        run_one dequant "$DEQUANT" "$PROMPT" "$GENTOK" "$DFILE" > /dev/null 2>&1
        DSTATS=$(extract_stats "$DFILE")

        BPTOK=$(echo $BSTATS | cut -d, -f1)
        BPRATE=$(echo $BSTATS | cut -d, -f2)
        BDRATE=$(echo $BSTATS | cut -d, -f4)
        BTTFT=$(echo $BSTATS | cut -d, -f5)
        DPRATE=$(echo $DSTATS | cut -d, -f2)
        DDRATE=$(echo $DSTATS | cut -d, -f4)
        DTTFT=$(echo $DSTATS | cut -d, -f5)

        echo "$BPTOK,$GENTOK,$BPRATE,$BDRATE,$BTTFT,$DPRATE,$DDRATE,$DTTFT" >> "$REPORT_DIR/sweep.csv"
        echo "  P=$BPTOK: baseline prefill=${BPRATE} decode=${BDRATE} | dequant prefill=${DPRATE} decode=${DDRATE}"
    done
done

echo ""
echo "Sweep complete. Results in $REPORT_DIR/sweep.csv"
