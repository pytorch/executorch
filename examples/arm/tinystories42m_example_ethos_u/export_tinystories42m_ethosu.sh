#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

artifact_dir="${TINYSTORIES42M_ARTIFACT_DIR:-${repo_root}/data/tinystories42m}"
checkpoint="${artifact_dir}/stories42M.pt"
params="${artifact_dir}/params.json"
tokenizer="${artifact_dir}/tokenizer.model"
output_dir="${artifact_dir}"
output_name="tinystories42m_ethosu_u85_256_ctx64_kv_w8a16.pte"
calibration_limit=1
calibration_seq_length=64
max_seq_length=64
max_context_length=64
target="ethos-u85-256"
system_config="Ethos_U85_SYS_DRAM_Mid"
memory_mode="Sram_Only"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --checkpoint=PATH              TinyStories-42M checkpoint.
  --params=PATH                  Model params JSON.
  --tokenizer=PATH               SentencePiece tokenizer.model.
  --output_dir=DIR               Export directory. Default: ${output_dir}
  --output_name=NAME             PTE filename. Default: ${output_name}
  --calibration_limit=N          Wikitext calibration sample count. Default: ${calibration_limit}
  --calibration_seq_length=N     PT2E calibration length. Default: ${calibration_seq_length}
  --max_seq_length=N             Export sequence capacity. Default: ${max_seq_length}
  --max_context_length=N         KV-cache capacity. Default: ${max_context_length}
  --target=NAME                  Ethos-U target. Default: ${target}
  --system_config=NAME           Vela system config. Default: ${system_config}
  --memory_mode=NAME             Vela memory mode. Default: ${memory_mode}
EOF
}

for arg in "$@"; do
  case "${arg}" in
    -h|--help) usage; exit 0 ;;
    --checkpoint=*) checkpoint="${arg#*=}" ;;
    --params=*) params="${arg#*=}" ;;
    --tokenizer=*) tokenizer="${arg#*=}" ;;
    --output_dir=*) output_dir="${arg#*=}" ;;
    --output_name=*) output_name="${arg#*=}" ;;
    --calibration_limit=*) calibration_limit="${arg#*=}" ;;
    --calibration_seq_length=*) calibration_seq_length="${arg#*=}" ;;
    --max_seq_length=*) max_seq_length="${arg#*=}" ;;
    --max_context_length=*) max_context_length="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --system_config=*) system_config="${arg#*=}" ;;
    --memory_mode=*) memory_mode="${arg#*=}" ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

for path in "${checkpoint}" "${params}" "${tokenizer}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required model file: ${path}" >&2
    exit 1
  fi
done

if [[ "${output_name}" != *.pte ]]; then
  output_name="${output_name}.pte"
fi
if [[ "${output_name}" = /* ]]; then
  output_path="${output_name}"
else
  output_path="${output_dir}/${output_name}"
fi
mkdir -p "${output_dir}" "$(dirname -- "${output_path}")"
cd "${repo_root}"

python -m extension.llm.export.export_llm \
  base.model_class=stories110m \
  base.checkpoint="${checkpoint}" \
  base.params="${params}" \
  base.tokenizer_path="${tokenizer}" \
  export.output_dir="${output_dir}" \
  export.output_name="${output_path}" \
  export.max_seq_length="${max_seq_length}" \
  export.max_context_length="${max_context_length}" \
  quantization.pt2e_quantize=ethosu_16a8w \
  quantization.quantize_scope=linear \
  quantization.calibration_tasks="[wikitext]" \
  quantization.calibration_limit="${calibration_limit}" \
  quantization.calibration_seq_length="${calibration_seq_length}" \
  backend.ethosu.enabled=True \
  backend.ethosu.target="${target}" \
  backend.ethosu.system_config="${system_config}" \
  backend.ethosu.memory_mode="${memory_mode}" \
  model.use_kv_cache=True \
  model.quantize_kv_cache=False \
  model.enable_dynamic_shape=False \
  debug.verbose=True \
  debug.generate_full_logits=False
