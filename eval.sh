#!/bin/bash

# 用法: ./evaluate_kernel.sh <kernel_name> <n_evaluation>
KERNEL_NAME=$1
N_EVAL=$2

# 路径前缀
BASE_PATH=~/kernel-gen/whisper-large-v3-turbo/${KERNEL_NAME}/

MODEL_PATH=${BASE_PATH}model.pte
DATA_PATH=${BASE_PATH}aoti_cuda_blob.ptd
TOKENIZER_PATH=${BASE_PATH}
AUDIO_PATH=${BASE_PATH}output.wav
PROCESSOR_PATH=${BASE_PATH}whisper_preprocessor.pte

CMD="cmake-out/examples/models/whisper/whisper_runner \
  --model_path ${MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --temperature 0 \
  --tokenizer_path ${TOKENIZER_PATH} \
  --audio_path ${AUDIO_PATH} \
  --processor_path ${PROCESSOR_PATH}"

rates=()
for ((i=1; i<=N_EVAL; i++)); do
  echo "Running evaluation $i/$N_EVAL..."
  output=$($CMD 2>&1)
  # 推荐用 awk
  rate=$(echo "$output" | grep "Generated 128 tokens:" | awk '{print $(NF-1)}')
  echo "Generated token rate for run $i: $rate"
  if [[ ! -z "$rate" ]]; then
    rates+=($rate)
  fi
done

# 计算平均值
sum=0
count=0
for r in "${rates[@]}"; do
  # 只统计非空数值
  if [[ ! -z "$r" ]]; then
    sum=$(echo "$sum + $r" | bc)
    count=$((count+1))
  fi
done

if [[ $count -gt 0 ]]; then
  avg=$(echo "scale=2; $sum / $count" | bc)
  echo "Average Generated token rate over $count runs: $avg tokens/second"
else
  echo "No valid token rates found."
fi
