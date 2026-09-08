#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_vulkan_acceptance.sh MODEL_DIR AUDIO_WAV OUTPUT_DIR [RUNNER]

Runs the full Voxtral Realtime Vulkan acceptance on an NVIDIA Linux host.
MODEL_DIR must contain the pinned Voxtral-Mini-4B-Realtime-2602 checkpoint.
AUDIO_WAV must be the pinned poem.wav acceptance fixture.
OUTPUT_DIR must be absent or empty. RUNNER defaults to the standard CMake
output and is rebuilt from the current source unless an explicit runner is
provided.

Environment overrides:
  PYTHON_EXECUTABLE                 Python command (default: python3)
  VOXTRAL_SKIP_EXPORT              Reuse complete artifacts in OUTPUT_DIR (0)
  VOXTRAL_FORCE_BUILD              Rebuild the default runner (1)
  VOXTRAL_NORMAL_PRIORITY          Run phases without nice/idle-I/O priority (0)
  VOXTRAL_PROFILE_METHODS          Pass --profile_methods to the runner (0)
  VOXTRAL_VULKAN_FORCE_FP16        Export with --vulkan-force-fp16 (0)
  VOXTRAL_STREAMING_ENCODER_BATCH_CHUNKS Streaming encoder chunks per call (1)
  VOXTRAL_THREADS                  Build/export thread limit (8)
  VOXTRAL_MAX_VMEM_GIB             Per-export virtual-memory cap (48)
  VOXTRAL_MIN_START_MEM_GIB        Required RAM before each phase (64)
  VOXTRAL_MIN_RUNTIME_MEM_GIB      Kill threshold while a phase runs (48)
  VOXTRAL_MIN_START_DISK_GIB       Required output-disk space (12)
  VOXTRAL_MIN_RUNTIME_DISK_GIB     Kill threshold while a phase runs (4)
  VOXTRAL_MIN_START_GPU_MIB        Required free GPU memory (65536)
  VOXTRAL_MIN_RUNTIME_GPU_MIB      Kill threshold while a phase runs (24576)
  VOXTRAL_MAX_START_LOAD_PERCENT   Maximum 1-minute load per online CPU (100)
  VOXTRAL_MAX_START_MEM_PSI_PERCENT Maximum memory PSI some/avg10 (5)
  VOXTRAL_MAX_SWAPOUT_PAGES        Kill threshold for new swap-out pages (262144)
  VOXTRAL_MONITOR_INTERVAL_SECONDS Resource sample interval (5)
  VOXTRAL_PHASE_TIMEOUT_SECONDS    Per-phase timeout (10800)
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

[[ $# -ge 3 && $# -le 4 ]] || {
  usage >&2
  exit 2
}

for command in awk cmake df find free grep ionice nproc nvidia-smi setsid sha256sum sort stat timeout uptime; do
  command -v "${command}" >/dev/null || fail "required command not found: ${command}"
done
[[ -x /usr/bin/time ]] || fail "required command not found: /usr/bin/time"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
model_dir=$(realpath -- "$1")
audio_path=$(realpath -- "$2")
output_dir=$(realpath -m -- "$3")
runner_was_explicit=0
if [[ $# -eq 4 ]]; then
  runner_path=$(realpath -m -- "$4")
  runner_was_explicit=1
else
  runner_path="${repo_root}/cmake-out/examples/models/voxtral_realtime/voxtral_realtime_runner"
fi

python_bin=${PYTHON_EXECUTABLE:-python3}
skip_export=${VOXTRAL_SKIP_EXPORT:-0}
force_build=${VOXTRAL_FORCE_BUILD:-1}
normal_priority=${VOXTRAL_NORMAL_PRIORITY:-0}
profile_methods=${VOXTRAL_PROFILE_METHODS:-0}
vulkan_force_fp16=${VOXTRAL_VULKAN_FORCE_FP16:-0}
streaming_encoder_batch_chunks=${VOXTRAL_STREAMING_ENCODER_BATCH_CHUNKS:-1}
thread_count=${VOXTRAL_THREADS:-8}
max_vmem_gib=${VOXTRAL_MAX_VMEM_GIB:-48}
min_start_mem_gib=${VOXTRAL_MIN_START_MEM_GIB:-64}
min_runtime_mem_gib=${VOXTRAL_MIN_RUNTIME_MEM_GIB:-48}
min_start_disk_gib=${VOXTRAL_MIN_START_DISK_GIB:-12}
min_runtime_disk_gib=${VOXTRAL_MIN_RUNTIME_DISK_GIB:-4}
min_start_gpu_mib=${VOXTRAL_MIN_START_GPU_MIB:-65536}
min_runtime_gpu_mib=${VOXTRAL_MIN_RUNTIME_GPU_MIB:-24576}
max_start_load_percent=${VOXTRAL_MAX_START_LOAD_PERCENT:-100}
max_start_mem_psi_percent=${VOXTRAL_MAX_START_MEM_PSI_PERCENT:-5}
max_swapout_pages=${VOXTRAL_MAX_SWAPOUT_PAGES:-262144}
monitor_interval=${VOXTRAL_MONITOR_INTERVAL_SECONDS:-5}
phase_timeout=${VOXTRAL_PHASE_TIMEOUT_SECONDS:-10800}

for value in \
  "$skip_export" \
  "$force_build" \
  "$normal_priority" \
  "$profile_methods" \
  "$vulkan_force_fp16" \
  "$streaming_encoder_batch_chunks" \
  "$thread_count" \
  "$max_vmem_gib" \
  "$min_start_mem_gib" \
  "$min_runtime_mem_gib" \
  "$min_start_disk_gib" \
  "$min_runtime_disk_gib" \
  "$min_start_gpu_mib" \
  "$min_runtime_gpu_mib" \
  "$max_start_load_percent" \
  "$max_start_mem_psi_percent" \
  "$max_swapout_pages" \
  "$monitor_interval" \
  "$phase_timeout"; do
  [[ $value =~ ^[0-9]+$ ]] || fail "resource settings must be non-negative integers"
done
[[ $skip_export == 0 || $skip_export == 1 ]] || fail "VOXTRAL_SKIP_EXPORT must be 0 or 1"
[[ $force_build == 0 || $force_build == 1 ]] || fail "VOXTRAL_FORCE_BUILD must be 0 or 1"
[[ $normal_priority == 0 || $normal_priority == 1 ]] || \
  fail "VOXTRAL_NORMAL_PRIORITY must be 0 or 1"
[[ $profile_methods == 0 || $profile_methods == 1 ]] || \
  fail "VOXTRAL_PROFILE_METHODS must be 0 or 1"
[[ $vulkan_force_fp16 == 0 || $vulkan_force_fp16 == 1 ]] || \
  fail "VOXTRAL_VULKAN_FORCE_FP16 must be 0 or 1"
[[ $streaming_encoder_batch_chunks == 1 || $streaming_encoder_batch_chunks == 2 ]] || \
  fail "VOXTRAL_STREAMING_ENCODER_BATCH_CHUNKS must be 1 or 2"
[[ $thread_count -gt 0 && $monitor_interval -gt 0 && $phase_timeout -gt 0 ]] || \
  fail "thread count, monitor interval, and phase timeout must be positive"
[[ $max_start_load_percent -gt 0 ]] || \
  fail "VOXTRAL_MAX_START_LOAD_PERCENT must be positive"

command -v "$python_bin" >/dev/null || fail "Python command not found: ${python_bin}"
[[ -d $model_dir ]] || fail "model directory not found: ${model_dir}"
[[ -f $audio_path ]] || fail "audio file not found: ${audio_path}"

params_path="${model_dir}/params.json"
weights_path="${model_dir}/consolidated.safetensors"
tokenizer_path="${model_dir}/tekken.json"
[[ -f $params_path && -f $weights_path && -f $tokenizer_path ]] || \
  fail "MODEL_DIR must contain params.json, consolidated.safetensors, and tekken.json"

[[ $(stat -c %s "$weights_path") == 8859462744 ]] || \
  fail "unexpected consolidated.safetensors size; expected pinned revision 2769294da9567371363522aac9bbcfdd19447add"
params_sha256=$(sha256sum "$params_path" | awk '{print $1}')
tokenizer_sha256=$(sha256sum "$tokenizer_path" | awk '{print $1}')
audio_sha256=$(sha256sum "$audio_path" | awk '{print $1}')
[[ $params_sha256 == 2ace010ebf7f0b62c60747d91c6d140e3c7238632d3e9c63d60a2bd2065ea301 ]] || \
  fail "params.json does not match the pinned checkpoint"
[[ $tokenizer_sha256 == 8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44 ]] || \
  fail "tekken.json does not match the pinned checkpoint"
[[ $audio_sha256 == 0dd03dfb6fe83b7d10df166cb77d28bf139f9be2c739e9927c757d88255aa88b ]] || \
  fail "AUDIO_WAV does not match the pinned poem.wav acceptance fixture"
weights_sha256=$(sha256sum "$weights_path" | awk '{print $1}')
[[ $weights_sha256 == 263f178fe752c90a2ae58f037a95ed092db8b14768b0978b8c48f66979c8345d ]] || \
  fail "consolidated.safetensors does not match the pinned checkpoint"

"$python_bin" - "$weights_path" <<'PY'
import sys

from safetensors import safe_open

with safe_open(sys.argv[1], framework="pt", device="cpu") as checkpoint:
    tensor_count = len(checkpoint.keys())
if tensor_count != 711:
    raise SystemExit(f"expected 711 checkpoint tensors, found {tensor_count}")
PY

if [[ -d $output_dir ]] && [[ -n $(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit) ]] && [[ $skip_export == 0 ]]; then
  fail "OUTPUT_DIR exists and is not empty: ${output_dir}"
fi
mkdir -p "$output_dir" "$output_dir/logs" "$output_dir/offline" "$output_dir/streaming"

mem_available_kib() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

disk_available_kib() {
  df -Pk "$output_dir" | awk 'NR == 2 {print $4}'
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
    awk 'NR == 1 {minimum=$1} $1 < minimum {minimum=$1} END {print int(minimum)}'
}

swapout_pages() {
  awk '/^pswpout / {print $2}' /proc/vmstat
}

load_1m_hundredths() {
  awk '{printf "%.0f\n", $1 * 100}' /proc/loadavg
}

memory_psi_hundredths() {
  awk '
    /^some / {
      for (i = 1; i <= NF; ++i) {
        if ($i ~ /^avg10=/) {
          split($i, value, "=")
          printf "%.0f\n", value[2] * 100
        }
      }
    }
  ' /proc/pressure/memory
}

check_start_resources() {
  local available_mem available_disk available_gpu load_1m mem_psi cpu_count
  available_mem=$(mem_available_kib)
  available_disk=$(disk_available_kib)
  available_gpu=$(gpu_free_mib)
  load_1m=$(load_1m_hundredths)
  mem_psi=$(memory_psi_hundredths)
  cpu_count=$(nproc)
  ((available_mem >= min_start_mem_gib * 1024 * 1024)) || \
    fail "only $((available_mem / 1024 / 1024)) GiB RAM available; need ${min_start_mem_gib} GiB"
  ((available_disk >= min_start_disk_gib * 1024 * 1024)) || \
    fail "only $((available_disk / 1024 / 1024)) GiB output-disk space available; need ${min_start_disk_gib} GiB"
  ((available_gpu >= min_start_gpu_mib)) || \
    fail "only ${available_gpu} MiB GPU memory available; need ${min_start_gpu_mib} MiB"
  ((load_1m <= cpu_count * max_start_load_percent)) || \
    fail "1-minute CPU load is $((load_1m / 100)); exceeds ${max_start_load_percent}% per online CPU"
  ((mem_psi <= max_start_mem_psi_percent * 100)) || \
    fail "memory PSI some/avg10 exceeds ${max_start_mem_psi_percent}%"
}

check_phase_resources() {
  check_start_resources
  uptime
  free -h
}

terminate_group() {
  local process_group=$1
  kill -TERM -- "-${process_group}" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 "$process_group" 2>/dev/null || return
    sleep 1
  done
  kill -KILL -- "-${process_group}" 2>/dev/null || true
}

active_process_group=""
cleanup_active_process_group() {
  if [[ -n $active_process_group ]]; then
    terminate_group "$active_process_group"
  fi
}
trap cleanup_active_process_group EXIT
trap 'exit 130' HUP INT TERM

run_guarded() {
  local label=$1
  local vmem_gib=$2
  shift 2
  check_phase_resources

  local log_path="${output_dir}/logs/${label}.log"
  local resource_path="${output_dir}/logs/${label}.resource.txt"
  local samples_path="${output_dir}/logs/${label}.samples.tsv"
  local initial_swapout process_id guard_reason=""
  initial_swapout=$(swapout_pages)
  printf 'epoch_seconds\tmem_available_kib\tdisk_available_kib\tgpu_free_mib\tload_1m_hundredths\tmemory_psi_hundredths\tnew_swapout_pages\n' >"$samples_path"

  (
    ulimit -c 0
    ulimit -v "$((vmem_gib * 1024 * 1024))"
    export OMP_NUM_THREADS="$thread_count"
    export MKL_NUM_THREADS="$thread_count"
    export OPENBLAS_NUM_THREADS="$thread_count"
    export CMAKE_BUILD_PARALLEL_LEVEL="$thread_count"
    if [[ $normal_priority == 1 ]]; then
      exec setsid -w \
        timeout --signal=TERM --kill-after=30s "$phase_timeout" \
        /usr/bin/time -v -o "$resource_path" "$@"
    fi
    exec setsid -w nice -n 10 ionice -c 3 \
      timeout --signal=TERM --kill-after=30s "$phase_timeout" \
      /usr/bin/time -v -o "$resource_path" "$@"
  ) >"$log_path" 2>&1 &
  process_id=$!
  active_process_group=$process_id

  while kill -0 "$process_id" 2>/dev/null; do
    local available_mem available_disk available_gpu current_swapout new_swapout load_1m mem_psi
    available_mem=$(mem_available_kib)
    available_disk=$(disk_available_kib)
    available_gpu=$(gpu_free_mib)
    load_1m=$(load_1m_hundredths)
    mem_psi=$(memory_psi_hundredths)
    current_swapout=$(swapout_pages)
    new_swapout=$((current_swapout - initial_swapout))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%s)" "$available_mem" "$available_disk" "$available_gpu" \
      "$load_1m" "$mem_psi" "$new_swapout" \
      >>"$samples_path"

    if ((available_mem < min_runtime_mem_gib * 1024 * 1024)); then
      guard_reason="RAM fell below ${min_runtime_mem_gib} GiB"
    elif ((available_disk < min_runtime_disk_gib * 1024 * 1024)); then
      guard_reason="output-disk space fell below ${min_runtime_disk_gib} GiB"
    elif ((available_gpu < min_runtime_gpu_mib)); then
      guard_reason="GPU memory fell below ${min_runtime_gpu_mib} MiB"
    elif ((new_swapout > max_swapout_pages)); then
      guard_reason="system swap-out exceeded ${max_swapout_pages} pages"
    fi
    if [[ -n $guard_reason ]]; then
      echo "resource guard stopped ${label}: ${guard_reason}" | tee -a "$log_path" >&2
      terminate_group "$process_id"
      break
    fi
    sleep "$monitor_interval"
  done

  local run_rc
  set +e
  wait "$process_id"
  run_rc=$?
  set -e
  active_process_group=""
  if [[ -n $guard_reason ]]; then
    fail "${label} was terminated by the resource guard"
  fi
  if [[ $run_rc -ne 0 ]]; then
    tail -n 100 "$log_path" >&2
    fail "${label} failed with exit status ${run_rc}"
  fi
}

if command -v vulkaninfo >/dev/null; then
  vulkan_summary=$(vulkaninfo --summary)
  printf '%s\n' "$vulkan_summary" | \
    grep -Eiq 'deviceName[[:space:]]*=[[:space:]]*NVIDIA' || \
    fail "Vulkan did not select an NVIDIA device"
else
  nvidia_icd=$(find /usr/share/vulkan/icd.d /etc/vulkan/icd.d \
    -maxdepth 1 -type f -iname '*nvidia*.json' -print -quit 2>/dev/null || true)
  [[ -n $nvidia_icd ]] || fail "vulkaninfo is unavailable and no NVIDIA Vulkan ICD was found"
  vulkan_summary="vulkaninfo unavailable; NVIDIA ICD=${nvidia_icd}"
fi

hardware_path="${output_dir}/hardware.txt"
{
  date --iso-8601=seconds
  git -C "$repo_root" rev-parse HEAD
  git -C "$repo_root" status --short
  uname -a
  uptime
  free -h
  df -h "$output_dir"
  printf 'streaming_encoder_batch_chunks=%s\n' "$streaming_encoder_batch_chunks"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
  printf '%s\n' "$vulkan_summary"
} >"$hardware_path"

check_start_resources

if [[ $runner_was_explicit == 0 && (! -x $runner_path || $force_build == 1) ]]; then
  run_guarded configure-executorch "$max_vmem_gib" \
    cmake -E chdir "$repo_root" cmake --preset llm-debug-vulkan
  run_guarded build-executorch "$max_vmem_gib" \
    cmake -E chdir "$repo_root" \
    cmake --build --preset llm-debug-vulkan-install --parallel "$thread_count"
  run_guarded configure-runner "$max_vmem_gib" \
    cmake -E chdir "${repo_root}/examples/models/voxtral_realtime" \
    cmake --preset voxtral-realtime-vulkan
  run_guarded build-runner "$max_vmem_gib" \
    cmake -E chdir "${repo_root}/examples/models/voxtral_realtime" \
    cmake --build --preset voxtral-realtime-vulkan --parallel "$thread_count"
fi
[[ -x $runner_path ]] || fail "runner not found or not executable: ${runner_path}"

offline_dir="${output_dir}/offline"
streaming_dir="${output_dir}/streaming"

if [[ $skip_export == 0 ]]; then
  run_guarded preprocessor-offline 16 \
    "$python_bin" -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --max_audio_len 300 \
    --output_file "${offline_dir}/preprocessor.pte"

  run_guarded preprocessor-streaming 16 \
    "$python_bin" -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --streaming \
    --output_file "${streaming_dir}/preprocessor.pte"

  export_common=(
    "$python_bin" -u "${repo_root}/examples/models/voxtral_realtime/export_voxtral_rt.py"
    --model-path "$model_dir"
    --backend vulkan
    --max-seq-len 4096
    --delay-tokens 6
    --qlinear 8da4w
    --qlinear-group-size 32
    --qlinear-encoder 8da4w
    --qlinear-encoder-group-size 32
    --qembedding 4w
    --qembedding-group-size 32
    --dtype fp32
  )
  if [[ $vulkan_force_fp16 == 1 ]]; then
    export_common+=(--vulkan-force-fp16)
  fi

  run_guarded export-offline "$max_vmem_gib" \
    "${export_common[@]}" \
    --output-dir "$offline_dir"

  run_guarded export-streaming "$max_vmem_gib" \
    "${export_common[@]}" \
    --output-dir "$streaming_dir" \
    --streaming \
    --streaming-encoder-batch-chunks "$streaming_encoder_batch_chunks" \
    --max-enc-len 750 \
    --sliding-window 8192
fi

ptd_list() {
  local artifact_dir=$1
  local paths=()
  while IFS= read -r -d '' path; do
    paths+=("$path")
  done < <(find -L "$artifact_dir" -maxdepth 1 -type f -name '*.ptd' -print0 | sort -z)
  [[ ${#paths[@]} -eq 4 ]] || fail "expected four PTDs in ${artifact_dir}, found ${#paths[@]}"
  local IFS=,
  printf '%s' "${paths[*]}"
}

[[ -f ${offline_dir}/model.pte && -f ${offline_dir}/preprocessor.pte ]] || \
  fail "offline artifacts are incomplete"
[[ -f ${streaming_dir}/model.pte && -f ${streaming_dir}/preprocessor.pte ]] || \
  fail "streaming artifacts are incomplete"
offline_ptds=$(ptd_list "$offline_dir")
streaming_ptds=$(ptd_list "$streaming_dir")
runner_profile_args=()
if [[ $profile_methods == 1 ]]; then
  runner_profile_args+=(--profile_methods)
fi

run_guarded run-offline 16 \
  "$runner_path" \
  --model_path "${offline_dir}/model.pte" \
  --data_paths "$offline_ptds" \
  --tokenizer_path "$tokenizer_path" \
  --preprocessor_path "${offline_dir}/preprocessor.pte" \
  --audio_path "$audio_path" \
  --offline_max_new_tokens 32 \
  "${runner_profile_args[@]}"

run_guarded run-streaming 16 \
  "$runner_path" \
  --model_path "${streaming_dir}/model.pte" \
  --data_paths "$streaming_ptds" \
  --tokenizer_path "$tokenizer_path" \
  --preprocessor_path "${streaming_dir}/preprocessor.pte" \
  --audio_path "$audio_path" \
  --streaming \
  "${runner_profile_args[@]}"

transcript_matches() {
  local log_path=$1
  local transcript=$2
  sed -E \
    -e 's/[DIEWF] [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+ executorch:.*$//' \
    -e 's/(Debug|Info|Warning|Error) \(XNNPACK\):.*$//' \
    "$log_path" | tr -d '\n' | grep -F "$transcript" >/dev/null
}

expected_transcript='Dancing in the masquerade, idle truth in plain sight jaded, pop, roll, click, shot, who will I be today or not? But such a tide as moving seems asleep, too full for sound and foam, when that which drew from out the boundless deep turns again home, twilight and evening bell and after that.'
transcript_matches "${output_dir}/logs/run-offline.log" "$expected_transcript" || \
  fail "offline transcript did not match the accepted fixture"
transcript_matches "${output_dir}/logs/run-streaming.log" "$expected_transcript" || \
  fail "streaming transcript did not match the accepted fixture"
grep -Fq 'Generated Tokens: 378' "${output_dir}/logs/run-offline.log" || \
  fail "offline generated-token count was not 378"
grep -Fq 'Generated Tokens: 306' "${output_dir}/logs/run-streaming.log" || \
  fail "streaming generated-token count was not 306"
grep -Fq "encoder_batch_chunks=${streaming_encoder_batch_chunks}" \
  "${output_dir}/logs/run-streaming.log" || \
  fail "streaming artifact did not use encoder_batch_chunks=${streaming_encoder_batch_chunks}"

(
  cd "$output_dir"
  find -L offline streaming -maxdepth 1 -type f -print0 | sort -z | \
    while IFS= read -r -d '' path; do
      sha256sum "$path"
    done
  printf '%s  runner\n' "$(sha256sum "$runner_path" | awk '{print $1}')"
  printf '%s  fixture/poem.wav\n' "$audio_sha256"
  printf '%s  checkpoint/params.json\n' "$params_sha256"
  printf '%s  checkpoint/tekken.json\n' "$tokenizer_sha256"
  printf '%s  checkpoint/consolidated.safetensors\n' "$weights_sha256"
) >"${output_dir}/artifacts.sha256"

echo "Voxtral Realtime Vulkan acceptance passed."
echo "Evidence: ${output_dir}"
