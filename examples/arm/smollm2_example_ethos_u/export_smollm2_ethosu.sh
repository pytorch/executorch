#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

mode="all"
quantize_scope="linear"
output_dir="${repo_root}"
tokenizer_path="${repo_root}/data/tokenizers/smollm2/tokenizer.json"
max_seq_length=32
max_context_length=32
calibration_limit=4
calibration_seq_length=32
target="ethos-u85-256"
system_config="Ethos_U85_SYS_DRAM_High"
memory_mode="Dedicated_Sram_512KB"
full_logits=1
ethosu_extra_flags=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode=all|w8a8|w8a16          Which export(s) to generate. Default: ${mode}
  --quantize_scope=full|linear   Arm PT2E quantization scope. Default: ${quantize_scope}
  --output_dir=DIR               Output directory. Default: ${output_dir}
                                 The repo-root default matches the quickstart.
  --tokenizer=PATH               Tokenizer JSON path. Default: ${tokenizer_path}
  --max_seq_length=N             Export window size. Default: ${max_seq_length}
  --max_context_length=N         Export context size. Default: ${max_context_length}
  --calibration_limit=N          Wikitext sample count. Default: ${calibration_limit}
  --calibration_seq_length=N     Calibration token window. Default: ${calibration_seq_length}
  --target=NAME                  Ethos-U target. Default: ${target}
  --system_config=NAME           Vela system config. Default: ${system_config}
  --memory_mode=NAME             Vela memory mode. Default: ${memory_mode}
  --ethosu_extra_flags=LIST      JSON-style Hydra list of extra Vela flags, e.g.
                                 '["--arena-cache-size=1048576"]'
  --full_logits                  Export full logits and append _full_logits to filenames.
                                 This is the default for calibrated static
                                 non-KV exports.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --mode=*) mode="${arg#*=}" ;;
    --quantize_scope=*) quantize_scope="${arg#*=}" ;;
    --output_dir=*) output_dir="${arg#*=}" ;;
    --tokenizer=*) tokenizer_path="${arg#*=}" ;;
    --max_seq_length=*) max_seq_length="${arg#*=}" ;;
    --max_context_length=*) max_context_length="${arg#*=}" ;;
    --calibration_limit=*) calibration_limit="${arg#*=}" ;;
    --calibration_seq_length=*) calibration_seq_length="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --system_config=*) system_config="${arg#*=}" ;;
    --memory_mode=*) memory_mode="${arg#*=}" ;;
    --ethosu_extra_flags=*) ethosu_extra_flags="${arg#*=}" ;;
    --full_logits) full_logits=1 ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "${output_dir}"

run_export() {
  local pt2e_quantize="$1"
  local output_name="$2"

  echo "[export] output_name=${output_name}"
  echo "[export] backend.ethosu.extra_flags=${ethosu_extra_flags:-[] }"

  local -a cmd=(
    python -m extension.llm.export.export_llm
    base.model_class=smollm2
    base.params=examples/models/smollm2/135M_config.json
    base.tokenizer_path="${tokenizer_path}"
    export.output_dir="${output_dir}"
    export.output_name="${output_name}"
    export.max_seq_length="${max_seq_length}"
    export.max_context_length="${max_context_length}"
    quantization.pt2e_quantize="${pt2e_quantize}"
    quantization.quantize_scope="${quantize_scope}"
    quantization.calibration_tasks="[wikitext]"
    quantization.calibration_limit="${calibration_limit}"
    quantization.calibration_seq_length="${calibration_seq_length}"
    backend.ethosu.enabled=True
    backend.ethosu.target="${target}"
    backend.ethosu.system_config="${system_config}"
    backend.ethosu.memory_mode="${memory_mode}"
    model.use_kv_cache=False
    model.enable_dynamic_shape=False
    debug.verbose=True
    debug.generate_full_logits=$( [[ "${full_logits}" -eq 1 ]] && echo True || echo False )
  )
  if [[ -n "${ethosu_extra_flags}" ]]; then
    cmd+=("backend.ethosu.extra_flags=${ethosu_extra_flags}")
  fi

  "${cmd[@]}"
}

output_name_for() {
  local stem="$1"
  if [[ "${full_logits}" -eq 1 ]]; then
    printf '%s_full_logits.pte' "${stem}"
  else
    printf '%s.pte' "${stem}"
  fi
}

cd "${repo_root}"

case "${mode}" in
  all)
    run_export ethosu_8a8w "$(output_name_for smollm2_ethosu_seq${max_seq_length}_w8a8_wikitext)"
    run_export ethosu_16a8w "$(output_name_for smollm2_ethosu_seq${max_seq_length}_w8a16_wikitext)"
    ;;
  w8a8)
    run_export ethosu_8a8w "$(output_name_for smollm2_ethosu_seq${max_seq_length}_w8a8_wikitext)"
    ;;
  w8a16)
    run_export ethosu_16a8w "$(output_name_for smollm2_ethosu_seq${max_seq_length}_w8a16_wikitext)"
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    exit 1
    ;;
esac
